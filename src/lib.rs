//! # waycap-rs
//!
//! `waycap-rs` is a high-level Wayland screen capture library with hardware-accelerated encoding.
//! It provides an easy-to-use API for capturing screen content on Wayland-based Linux systems,
//! using PipeWire for screen capture and hardware accelerated encoding for both video and audio.
//!
//! ## Features
//!
//! - Hardware-accelerated encoding (VAAPI and NVENC)
//! - No Copy approach to encoding video frames utilizing DMA Buffers
//! - Audio capture support
//! - Multiple quality presets
//! - Cursor visibility control
//! - Fine-grained control over capture (start, pause, resume)
//!
//! ## Platform Support
//!
//! This library currently supports Linux with Wayland display server and
//! requires the XDG Desktop Portal and PipeWire for screen capture.
//!
//! ## Example
//!
//! ```rust
//! use waycap_rs::{CaptureBuilder, QualityPreset, VideoEncoder, AudioEncoder};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a capture instance
//!     let mut capture = CaptureBuilder::new()
//!         .with_audio()
//!         .with_quality_preset(QualityPreset::Medium)
//!         .with_cursor_shown()
//!         .with_video_encoder(VideoEncoder::Vaapi)
//!         .with_audio_encoder(AudioEncoder::Opus)
//!         .build()?;
//!     
//!     // Start capturing
//!     capture.start()?;
//!     
//!     // Get receivers for encoded frames
//!     let video_receiver = capture.get_video_receiver();
//!     let audio_receiver = capture.get_audio_receiver()?;
//!     
//!     // Process frames as needed...
//!     
//!     // Stop capturing when done
//!     capture.close()?;
//!     
//!     Ok(())
//! }
//! ```

#![warn(clippy::all)]
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self},
        Arc,
    },
    time::{Duration, Instant},
};

use capture::{audio::AudioCapture, video::VideoCapture, Terminate};
use crossbeam::{
    channel::{bounded, Receiver, Sender},
    select,
};
use encoders::{audio::AudioEncoder, opus_encoder::OpusEncoder};
use portal_screencast_waycap::{CursorMode, ScreenCast, SourceType};
use std::sync::Mutex;
use types::{
    audio_frame::{EncodedAudioFrame, RawAudioFrame},
    config::{AudioEncoder as AudioEncoderType, QualityPreset, VideoEncoder as VideoEncoderType},
    error::{Result, WaycapError},
    video_frame::{EncodedVideoFrame, RawVideoFrame},
};
use waycap_egl::{EglContext, GpuVendor};

mod capture;
mod encoders;
pub mod pipeline;
pub mod types;
mod utils;
mod waycap_egl;

pub use encoders::video::VideoEncoder;
pub use utils::TIME_UNIT_NS;

pub use crate::encoders::dynamic_encoder::DynamicEncoder;
pub use crate::encoders::image_encoder::ImageEncoder;
use crate::encoders::video::{start_video_loop, PipewireSPA};

/// Target Screen Resolution
pub struct Resolution {
    width: u32,
    height: u32,
}

/// Main capture instance for recording screen content and audio.
///
/// `Capture` provides methods to control the recording process, retrieve
/// encoded frames, and manage the capture lifecycle.
///
/// # Examples
///
/// ```
/// use waycap_rs::{CaptureBuilder, QualityPreset, VideoEncoder};
///
/// // Create a capture instance
/// let mut capture = CaptureBuilder::new()
///     .with_quality_preset(QualityPreset::Medium)
///     .with_video_encoder(VideoEncoder::Vaapi)
///     .build()
///     .expect("Failed to create capture");
///
/// // Start the capture
/// capture.start().expect("Failed to start capture");
///
/// // Get video receiver
/// let video_receiver = capture.get_video_receiver();
///
/// // Process Frames
/// while let Some(encoded_frame) = video_receiver.try_pop() {
///     println!("Received an encoded frame");
/// }
pub struct Capture<V: VideoEncoder + PipewireSPA + Send> {
    stop_flag: Arc<AtomicBool>,
    pause_flag: Arc<AtomicBool>,

    worker_handles: Vec<std::thread::JoinHandle<Result<()>>>,

    video_encoder: Option<Arc<Mutex<V>>>,
    pw_video_terminate_tx: Option<pipewire::channel::Sender<Terminate>>,

    audio_encoder: Option<Arc<Mutex<dyn AudioEncoder + Send>>>,
    pw_audio_terminate_tx: Option<pipewire::channel::Sender<Terminate>>,
}

impl<V: VideoEncoder + PipewireSPA> Capture<V> {
    pub fn new_with_encoder(video_encoder: V, include_cursor: bool, target_fps: u64) -> Result<Self>
    where
        V: 'static,
    {
        let mut _self = Self {
            stop_flag: Arc::new(AtomicBool::new(false)),
            pause_flag: Arc::new(AtomicBool::new(false)),
            worker_handles: Vec::new(),
            video_encoder: None,
            audio_encoder: None,
            pw_video_terminate_tx: None,
            pw_audio_terminate_tx: None,
        };

        let (frame_rx, video_ready, audio_ready, resolution) =
            _self.start_pipewire_video(include_cursor)?;

        std::thread::sleep(Duration::from_millis(100));
        audio_ready.store(true, Ordering::Release);
        _self.start().unwrap();

        println!("waiting for video ready");
        // while !video_ready.load(Ordering::Acquire) {
        //     std::thread::sleep(Duration::from_millis(100));
        //     println!(".")
        // }

        let (video_encoder, video_loop) = start_video_loop(
            video_encoder,
            frame_rx,
            Arc::clone(&_self.stop_flag),
            Arc::clone(&_self.pause_flag),
            target_fps,
        );
        _self.video_encoder = Some(video_encoder);
        _self.worker_handles.push(video_loop);

        log::info!("Capture started successfully.");
        Ok(_self)
    }

    fn start_pipewire_video(
        &mut self,
        include_cursor: bool,
    ) -> Result<(
        Receiver<RawVideoFrame>,
        Arc<AtomicBool>,
        Arc<AtomicBool>,
        Resolution,
    )> {
        let (frame_tx, frame_rx): (Sender<RawVideoFrame>, Receiver<RawVideoFrame>) = bounded(10);

        let audio_ready = Arc::new(AtomicBool::new(false));
        let audio_ready_pw = Arc::clone(&audio_ready);

        let video_ready = Arc::new(AtomicBool::new(false));
        let video_ready_pw = Arc::clone(&video_ready);

        let (pw_sender, pw_recv) = pipewire::channel::channel();
        self.pw_video_terminate_tx = Some(pw_sender);

        let (reso_sender, reso_recv) = mpsc::channel::<Resolution>();

        let mut screen_cast = ScreenCast::new()?;
        screen_cast.set_source_types(SourceType::all());
        screen_cast.set_cursor_mode(if include_cursor {
            CursorMode::EMBEDDED
        } else {
            CursorMode::HIDDEN
        });
        let active_cast = screen_cast.start(None)?;
        let fd = active_cast.pipewire_fd();
        let stream = active_cast.streams().next().unwrap();
        let stream_node = stream.pipewire_node();
        let pause_video = Arc::clone(&self.pause_flag);
        self.worker_handles
            .push(std::thread::spawn(move || -> Result<()> {
                let mut video_cap = match VideoCapture::new(
                    fd,
                    stream_node,
                    video_ready_pw,
                    audio_ready_pw,
                    pause_video,
                    reso_sender,
                    frame_tx,
                    pw_recv,
                    V::get_spa_definition()?,
                ) {
                    Ok(pw_capture) => pw_capture,
                    Err(e) => {
                        log::error!("Error initializing pipewire struct: {e:}");
                        return Err(e);
                    }
                };

                video_cap.run()?;

                let _ = active_cast.close(); // Keep this alive until the thread ends
                Ok(())
            }));

        // Wait to get back a negotiated resolution from pipewire
        let timeout = Duration::from_secs(5);
        let start = Instant::now();
        let resolution = loop {
            if let Ok(reso) = reso_recv.recv() {
                break reso;
            }

            if start.elapsed() > timeout {
                log::error!("Timeout waiting for PipeWire negotiated resolution.");
                return Err(WaycapError::Init(
                    "Timed out waiting for pipewire to negotiate video resolution".into(),
                ));
            }

            std::thread::sleep(Duration::from_millis(100));
        };

        Ok((frame_rx, video_ready, audio_ready, resolution))
    }

    /// Enables capture streams to send their frames to their encoders
    pub fn start(&mut self) -> Result<()> {
        self.pause_flag.store(false, Ordering::Release);
        Ok(())
    }

    /// Temporarily stops the recording by blocking frames from being sent to the encoders
    pub fn pause(&mut self) -> Result<()> {
        self.pause_flag.store(true, Ordering::Release);
        Ok(())
    }

    /// Stop recording and drain the encoders of any last frames they have in their internal
    /// buffers. These frames are discarded.
    pub fn finish(&mut self) -> Result<()> {
        self.pause_flag.store(true, Ordering::Release);
        if let Some(ref mut enc) = self.video_encoder {
            enc.lock().unwrap().drain()?;
        }
        if let Some(ref mut enc) = self.audio_encoder {
            enc.lock().unwrap().drain()?;
        }
        Ok(())
    }

    /// Resets the encoder states so we can resume encoding from within this same session
    pub fn reset(&mut self) -> Result<()> {
        if let Some(ref mut enc) = self.video_encoder {
            enc.lock().unwrap().reset()?;
        }
        if let Some(ref mut enc) = self.audio_encoder {
            enc.lock().unwrap().reset()?;
        }

        Ok(())
    }

    /// Close the connection. Once called the struct cannot be re-used and must be re-built with
    /// the [`crate::pipeline::builder::CaptureBuilder`] to record again.
    /// If your goal is to temporarily stop recording use [`Self::pause`] or [`Self::finish`] + [`Self::reset`]
    pub fn close(&mut self) -> Result<()> {
        self.finish()?;
        self.stop_flag.store(true, Ordering::Release);
        if let Some(pw_vid) = &self.pw_video_terminate_tx {
            let _ = pw_vid.send(Terminate {});
        }
        if let Some(pw_aud) = &self.pw_audio_terminate_tx {
            let _ = pw_aud.send(Terminate {});
        }

        for handle in self.worker_handles.drain(..) {
            let _ = handle.join();
        }

        drop(self.video_encoder.take());
        drop(self.audio_encoder.take());

        Ok(())
    }

    pub fn get_output(&mut self) -> Receiver<V::Output> {
        self.video_encoder
            .as_mut()
            .unwrap()
            .lock()
            .unwrap()
            .output()
            .unwrap()
    }
}

impl Capture<DynamicEncoder> {
    fn start_pipewire_audio(
        &mut self,
        audio_encoder_type: AudioEncoderType,
        include_audio: bool,
        video_ready: &Arc<AtomicBool>,
        audio_ready: &Arc<AtomicBool>,
    ) -> Result<Receiver<RawAudioFrame>> {
        let (pw_audio_sender, pw_audio_recv) = pipewire::channel::channel();
        self.pw_audio_terminate_tx = Some(pw_audio_sender);
        let (audio_tx, audio_rx): (Sender<RawAudioFrame>, Receiver<RawAudioFrame>) = bounded(10);
        if include_audio {
            let pause_capture = Arc::clone(&self.pause_flag);
            let video_r = Arc::clone(video_ready);
            let audio_r = Arc::clone(audio_ready);
            let pw_audio_worker = std::thread::spawn(move || -> Result<()> {
                log::debug!("Starting audio stream");
                let audio_cap = AudioCapture::new(video_r, audio_r);
                audio_cap.run(audio_tx, pw_audio_recv, pause_capture)?;
                Ok(())
            });

            self.worker_handles.push(pw_audio_worker);

            let enc: Arc<Mutex<dyn AudioEncoder + Send>> = match audio_encoder_type {
                AudioEncoderType::Opus => Arc::new(Mutex::new(OpusEncoder::new()?)),
            };

            self.audio_encoder = Some(enc);
        } else {
            audio_ready.store(true, Ordering::Release);
        }
        Ok(audio_rx)
    }

    pub fn new(
        video_encoder_type: Option<VideoEncoderType>,
        audio_encoder_type: AudioEncoderType,
        quality: QualityPreset,
        include_cursor: bool,
        include_audio: bool,
        target_fps: u64,
    ) -> Result<Self> {
        let mut _self = Self {
            stop_flag: Arc::new(AtomicBool::new(false)),
            pause_flag: Arc::new(AtomicBool::new(true)),
            worker_handles: Vec::new(),
            video_encoder: None,
            audio_encoder: None,
            pw_video_terminate_tx: None,
            pw_audio_terminate_tx: None,
        };

        let encoder_type = resolve_video_encoder(video_encoder_type)?;
        let (frame_rx, video_ready, audio_ready, resolution) =
            _self.start_pipewire_video(include_cursor)?;

        let video_encoder =
            DynamicEncoder::new(encoder_type, resolution.width, resolution.height, quality)?;

        let audio_rx = _self.start_pipewire_audio(
            audio_encoder_type,
            include_audio,
            &video_ready,
            &audio_ready,
        )?;

        // Wait until both threads are ready
        while !audio_ready.load(Ordering::Acquire) || !video_ready.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(100));
        }
        if include_audio {
            let audio_loop = audio_encoding_loop(
                Arc::clone(_self.audio_encoder.as_ref().unwrap()),
                audio_rx,
                Arc::clone(&_self.stop_flag),
                Arc::clone(&_self.pause_flag),
            );

            _self.worker_handles.push(audio_loop);
        }

        let (video_encoder, video_loop) = start_video_loop(
            video_encoder,
            frame_rx,
            Arc::clone(&_self.stop_flag),
            Arc::clone(&_self.pause_flag),
            target_fps,
        );
        _self.video_encoder = Some(video_encoder);
        _self.worker_handles.push(video_loop);

        log::info!("Capture started successfully.");
        Ok(_self)
    }

    /// Get a channel for which to receive encoded video frames.
    ///
    /// Returns a [`crossbeam::channel::Receiver`] which allows multiple consumers.
    /// Each call creates a new consumer that will receive all future frames.
    pub fn get_video_receiver(&mut self) -> Receiver<EncodedVideoFrame> {
        self.video_encoder
            .as_mut()
            .expect("Cannot access a video encoder which was never started.")
            .lock()
            .unwrap()
            .output()
            .unwrap()
    }

    /// Get a channel for which to receive encoded audio frames.
    ///
    /// Returns a [`crossbeam::channel::Receiver`] which allows multiple consumers.
    /// Each call creates a new consumer that will receive all future frames.
    pub fn get_audio_receiver(&mut self) -> Result<Receiver<EncodedAudioFrame>> {
        if let Some(ref mut audio_enc) = self.audio_encoder {
            return Ok(audio_enc.lock().unwrap().get_encoded_recv().unwrap());
        } else {
            Err(WaycapError::Validation(
                "Audio encoder does not exist".to_string(),
            ))
        }
    }

    /// Perform an action with the video encoder
    /// # Examples
    ///
    /// ```
    /// let mut output = ffmpeg::format::output(&filename)?;
    ///
    /// capture.with_video_encoder(|enc| {
    ///     if let Some(video_encoder) = enc {
    ///         let mut video_stream = output.add_stream(video_encoder.codec().unwrap()).unwrap();
    ///         video_stream.set_time_base(video_encoder.time_base());
    ///         video_stream.set_parameters(video_encoder);
    ///     }
    /// });
    /// output.write_header()?;
    pub fn with_video_encoder<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Option<ffmpeg_next::encoder::Video>) -> R,
    {
        let guard = self
            .video_encoder
            .as_ref()
            .expect("Cannot access a video encoder which was never started.")
            .lock()
            .unwrap();
        f(guard.get_encoder())
    }

    /// Perform an action with the audio encoder
    /// # Examples
    ///
    /// ```
    /// let mut output = ffmpeg::format::output(&filename)?;
    /// capture.with_audio_encoder(|enc| {
    ///     if let Some(audio_encoder) = enc {
    ///         let mut audio_stream = output.add_stream(audio_encoder.codec().unwrap()).unwrap();
    ///         audio_stream.set_time_base(audio_encoder.time_base());
    ///         audio_stream.set_parameters(audio_encoder);
    ///
    ///     }
    /// });
    /// output.write_header()?;
    pub fn with_audio_encoder<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Option<ffmpeg_next::encoder::Audio>) -> R,
    {
        assert!(self.audio_encoder.is_some());

        let guard = self.audio_encoder.as_ref().unwrap().lock().unwrap();
        f(guard.get_encoder())
    }
}

fn resolve_video_encoder(video_encoder_type: Option<VideoEncoderType>) -> Result<VideoEncoderType> {
    let encoder_type = match video_encoder_type {
        Some(typ) => typ,
        None => {
            // Dummy dimensions we just use this go get GPU vendor then drop it
            let dummy_context = EglContext::new(100, 100)?;
            match dummy_context.get_gpu_vendor() {
                GpuVendor::NVIDIA => VideoEncoderType::H264Nvenc,
                GpuVendor::AMD | GpuVendor::INTEL => VideoEncoderType::H264Vaapi,
                GpuVendor::UNKNOWN => {
                    return Err(WaycapError::Init(
                        "Unknown/Unimplemented GPU vendor".to_string(),
                    ));
                }
            }
        }
    };
    Ok(encoder_type)
}

impl<V: VideoEncoder + PipewireSPA> Drop for Capture<V> {
    fn drop(&mut self) {
        let _ = self.close();

        for handle in self.worker_handles.drain(..) {
            let _ = handle.join();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn audio_encoding_loop(
    audio_encoder: Arc<Mutex<dyn AudioEncoder + Send>>,
    audio_recv: Receiver<RawAudioFrame>,
    stop: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
) -> std::thread::JoinHandle<Result<()>> {
    std::thread::spawn(move || -> Result<()> {
        // CUDA contexts are thread local so set ours to this thread

        while !stop.load(Ordering::Acquire) {
            if pause.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }

            select! {
                recv(audio_recv) -> raw_samples => {
                    match raw_samples {
                        Ok(raw_samples) => {
                            // If we are getting samples then we know this must be set or we
                            // wouldn't be in here
                            audio_encoder.as_ref().lock().unwrap().process(raw_samples)?;
                        }
                        Err(_) => {
                            log::info!("Audio channel disconnected");
                            break;
                        }
                    }
                }
                default(Duration::from_millis(100)) => {
                    // Timeout to check stop/pause flags periodically
                }
            }
        }
        Ok(())
    })
}
