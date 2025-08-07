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
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use capture::{audio::AudioCapture, video::VideoCapture, Terminate};
use crossbeam::{
    channel::{bounded, Receiver, Sender},
    select,
};
use encoders::{
    audio::AudioEncoder, nvenc_encoder::NvencEncoder, opus_encoder::OpusEncoder,
    vaapi_encoder::VaapiEncoder, video::VideoEncoder,
};
use khronos_egl::Image;
use portal_screencast_waycap::{CursorMode, ScreenCast, SourceType};
use types::{
    audio_frame::{EncodedAudioFrame, RawAudioFrame},
    config::{AudioEncoder as AudioEncoderType, QualityPreset, VideoEncoder as VideoEncoderType},
    error::{Result, WaycapError},
    video_frame::{EncodedVideoFrame, RawVideoFrame},
};
use utils::{calculate_dimensions, extract_dmabuf_planes};
use waycap_egl::{EglContext, GpuVendor};

use crate::utils::TIME_UNIT_NS;

mod capture;
mod encoders;
pub mod pipeline;
pub mod types;
mod utils;
mod waycap_egl;

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
pub struct Capture {
    video_encoder: Arc<Mutex<dyn VideoEncoder + Send>>,
    audio_encoder: Option<Arc<Mutex<dyn AudioEncoder + Send>>>,
    stop_flag: Arc<AtomicBool>,
    pause_flag: Arc<AtomicBool>,
    egl_ctx: Arc<EglContext>,

    worker_handles: Vec<std::thread::JoinHandle<Result<()>>>,

    pw_video_terminate_tx: pipewire::channel::Sender<Terminate>,
    pw_audio_terminate_tx: Option<pipewire::channel::Sender<Terminate>>,
}

impl Capture {
    fn new(
        video_encoder_type: Option<VideoEncoderType>,
        audio_encoder_type: AudioEncoderType,
        quality: QualityPreset,
        include_cursor: bool,
        include_audio: bool,
        target_fps: u64,
    ) -> Result<Self> {
        let pause = Arc::new(AtomicBool::new(true));
        let stop = Arc::new(AtomicBool::new(false));

        let mut join_handles = Vec::new();

        let audio_ready = Arc::new(AtomicBool::new(false));
        let video_ready = Arc::new(AtomicBool::new(false));

        let (frame_tx, frame_rx): (Sender<RawVideoFrame>, Receiver<RawVideoFrame>) = bounded(10);

        let (pw_sender, pw_recv) = pipewire::channel::channel();
        let (reso_sender, reso_recv) = mpsc::channel::<Resolution>();
        let video_ready_pw = Arc::clone(&video_ready);
        let audio_ready_pw = Arc::clone(&audio_ready);
        let pause_video = Arc::clone(&pause);

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

        let use_nvenc_modifiers = match encoder_type {
            VideoEncoderType::H264Nvenc => true,
            VideoEncoderType::H264Vaapi => false,
        };

        let pw_video_capure = std::thread::spawn(move || -> Result<()> {
            let mut video_cap = match VideoCapture::new(
                fd,
                stream_node,
                video_ready_pw,
                audio_ready_pw,
                use_nvenc_modifiers,
                pause_video,
                reso_sender,
                frame_tx,
                pw_recv,
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
        });

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

        join_handles.push(pw_video_capure);

        let egl_context = Arc::new(EglContext::new(
            resolution.width as i32,
            resolution.height as i32,
        )?);

        let video_encoder: Arc<Mutex<dyn VideoEncoder + Send>> = match encoder_type {
            VideoEncoderType::H264Nvenc => {
                let mut encoder = NvencEncoder::new(resolution.width, resolution.height, quality)?;
                egl_context.create_persistent_texture()?;
                encoder.init_gl(egl_context.get_texture_id().unwrap())?;

                Arc::new(Mutex::new(encoder))
            }
            VideoEncoderType::H264Vaapi => Arc::new(Mutex::new(VaapiEncoder::new(
                resolution.width,
                resolution.height,
                quality,
            )?)),
        };

        let mut audio_encoder: Option<Arc<Mutex<dyn AudioEncoder + Send>>> = None;
        let (pw_audio_sender, pw_audio_recv) = pipewire::channel::channel();
        let (audio_tx, audio_rx): (Sender<RawAudioFrame>, Receiver<RawAudioFrame>) = bounded(10);
        if include_audio {
            let pause_capture = Arc::clone(&pause);
            let video_r = Arc::clone(&video_ready);
            let audio_r = Arc::clone(&audio_ready);
            let pw_audio_worker = std::thread::spawn(move || -> Result<()> {
                log::debug!("Starting audio stream");
                let audio_cap = AudioCapture::new(video_r, audio_r);
                audio_cap.run(audio_tx, pw_audio_recv, pause_capture)?;
                Ok(())
            });

            join_handles.push(pw_audio_worker);

            let enc: Arc<Mutex<dyn AudioEncoder + Send>> = match audio_encoder_type {
                AudioEncoderType::Opus => Arc::new(Mutex::new(OpusEncoder::new()?)),
            };

            audio_encoder = Some(enc);
        } else {
            audio_ready.store(true, Ordering::Release);
        }

        // Wait until both threads are ready
        while !audio_ready.load(Ordering::Acquire) || !video_ready.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(100));
        }

        let encoding_loop = encoding_loop(
            Arc::clone(&video_encoder),
            if include_audio {
                Some(Arc::clone(audio_encoder.as_ref().unwrap()))
            } else {
                None
            },
            frame_rx,
            audio_rx,
            Arc::clone(&stop),
            Arc::clone(&pause),
            target_fps,
            Arc::clone(&egl_context),
        );

        join_handles.push(encoding_loop);

        log::info!("Capture started sucessfully.");

        Ok(Self {
            video_encoder,
            audio_encoder,
            stop_flag: stop,
            pause_flag: pause,
            worker_handles: join_handles,
            pw_video_terminate_tx: pw_sender,
            pw_audio_terminate_tx: Some(pw_audio_sender),
            egl_ctx: egl_context,
        })
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
        self.video_encoder.lock().unwrap().drain()?;
        if let Some(ref mut enc) = self.audio_encoder {
            enc.lock().unwrap().drain()?;
        }

        Ok(())
    }

    /// Resets the encoder states so we can resume encoding from within this same session
    pub fn reset(&mut self) -> Result<()> {
        self.video_encoder.lock().unwrap().reset()?;
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
        let _ = self.pw_video_terminate_tx.send(Terminate {});
        if let Some(pw_aud) = &self.pw_audio_terminate_tx {
            let _ = pw_aud.send(Terminate {});
        }

        for handle in self.worker_handles.drain(..) {
            let _ = handle.join();
        }

        self.video_encoder.lock().unwrap().drop_encoder();
        self.audio_encoder.take();

        Ok(())
    }

    /// Get a channel for which to receive encoded video frames.
    ///
    /// Returns a [`crossbeam::channel::Receiver`] which allows multiple consumers.
    /// Each call creates a new consumer that will receive all future frames.
    pub fn get_video_receiver(&mut self) -> Receiver<EncodedVideoFrame> {
        self.video_encoder
            .lock()
            .unwrap()
            .get_encoded_recv()
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
        let guard = self.video_encoder.lock().unwrap();
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

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.close();

        // Make OpenGL context current to this thread before we drop nvenc which relies on it
        let _ = self.egl_ctx.release_current();
        let _ = self.egl_ctx.make_current();

        for handle in self.worker_handles.drain(..) {
            let _ = handle.join();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encoding_loop(
    video_encoder: Arc<Mutex<dyn VideoEncoder + Send>>,
    audio_encoder: Option<Arc<Mutex<dyn AudioEncoder + Send>>>,
    video_recv: Receiver<RawVideoFrame>,
    audio_recv: Receiver<RawAudioFrame>,
    stop: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
    target_fps: u64,
    egl_context: Arc<EglContext>,
) -> std::thread::JoinHandle<Result<()>> {
    egl_context.release_current().unwrap();

    std::thread::spawn(move || -> Result<()> {
        // CUDA contexts are thread local so set ours to this thread
        let is_nvenc = video_encoder.lock().unwrap().as_any().is::<NvencEncoder>();
        if is_nvenc {
            video_encoder
                .lock()
                .unwrap()
                .as_any()
                .downcast_ref::<NvencEncoder>()
                .unwrap()
                .make_current()?;
        }
        egl_context.make_current()?;

        let mut last_timestamp: u64 = 0;
        let frame_interval = TIME_UNIT_NS / target_fps;

        while !stop.load(Ordering::Acquire) {
            if pause.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }

            if audio_encoder.is_some() {
                select! {
                    recv(video_recv) -> raw_frame => {
                        match raw_frame {
                            Ok(raw_frame) => {
                                let current_time = raw_frame.timestamp as u64;
                                if current_time >= last_timestamp + frame_interval {
                                    if is_nvenc {
                                        match process_dmabuf_frame(&egl_context, &raw_frame) {
                                            Ok(img) => {
                                                video_encoder.lock().unwrap().process(&raw_frame)?;
                                                egl_context.destroy_image(img)?;
                                            }
                                            Err(e) => log::error!("Could not process dma buf frame: {e:?}"),
                                        }
                                    } else {
                                        video_encoder.lock().unwrap().process(&raw_frame)?;
                                    }
                                    last_timestamp = current_time;
                                }
                            }
                            Err(_) => {
                                log::info!("Video channel disconnected");
                                break;
                            }
                        }
                    }
                    recv(audio_recv) -> raw_samples => {
                        match raw_samples {
                            Ok(raw_samples) => {
                                // If we are getting samples then we know this must be set or we
                                // wouldn't be in here
                                audio_encoder.as_ref().unwrap().lock().unwrap().process(raw_samples)?;
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
            } else {
                select! {
                    recv(video_recv) -> raw_frame => {
                        match raw_frame {
                            Ok(raw_frame) => {
                                let current_time = raw_frame.timestamp as u64;
                                if current_time >= last_timestamp + frame_interval {
                                    if is_nvenc {
                                        match process_dmabuf_frame(&egl_context, &raw_frame) {
                                            Ok(img) => {
                                                video_encoder.lock().unwrap().process(&raw_frame)?;
                                                egl_context.destroy_image(img)?;
                                            }
                                            Err(e) => log::error!("Could not process dma buf frame: {e:?}"),
                                        }
                                    } else {
                                        video_encoder.lock().unwrap().process(&raw_frame)?;
                                    }
                                    last_timestamp = current_time;
                                }
                            }
                            Err(_) => {
                                log::info!("Video channel disconnected");
                                break;
                            }
                        }
                    }
                    default(Duration::from_millis(100)) => {
                        // Timeout to check stop/pause flags periodically
                    }
                }
            }
        }
        Ok(())
    })
}

fn process_dmabuf_frame(egl_ctx: &EglContext, raw_frame: &RawVideoFrame) -> Result<Image> {
    let dma_buf_planes = extract_dmabuf_planes(raw_frame)?;

    let format = drm_fourcc::DrmFourcc::Argb8888 as u32;
    let (width, height) = calculate_dimensions(raw_frame)?;
    let modifier = raw_frame.modifier;

    let egl_image =
        egl_ctx.create_image_from_dmabuf(&dma_buf_planes, format, width, height, modifier)?;

    egl_ctx.update_texture_from_image(egl_image)?;

    Ok(egl_image)
}
