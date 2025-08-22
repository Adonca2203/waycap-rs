use std::ffi::CString;
use std::ptr::null_mut;
use std::sync::Arc;
use std::time::Duration;

use crate::types::error::{Result, WaycapError};
use crate::types::video_frame::RawVideoFrame;
use crate::CaptureControls;
use crossbeam::channel::Receiver;
use crossbeam::select;
use ffmpeg::ffi::{av_hwdevice_ctx_create, av_hwframe_ctx_alloc, AVBufferRef};
use ffmpeg_next::{self as ffmpeg};
use pipewire::spa;
use std::sync::Mutex;

pub const GOP_SIZE: u32 = 30;

/// Base trait for video encoders. defines the output type of an encoder.
///
/// To use this, implement either [`ProcessingThread::process`] for processing individual frames on
/// a separate worker thread, or [`StartVideoEncoder::start_processing`] for custom start logic.
pub trait VideoEncoder: Send + 'static {
    type Output;

    fn reset(&mut self) -> Result<()>;
    fn output(&mut self) -> Option<Receiver<Self::Output>>;
    fn drop_processor(&mut self);
    fn drain(&mut self) -> Result<()>;
    fn get_encoder(&self) -> &Option<ffmpeg::codec::encoder::Video>;
}

/// Specifies how processing is started for a encoder
/// For the default processing thread logic, implement [``ProcessingThread``] instead.
pub trait StartVideoEncoder: VideoEncoder + Sized {
    fn start_processing(
        capture: &mut crate::Capture<Self>,
        input: Receiver<RawVideoFrame>,
    ) -> Result<()>;
}

/// Implemented for all VideoEncoders which use a normal processing thread
///
/// [`ProcessingThread::process`] will be called with each frame
pub trait ProcessingThread: StartVideoEncoder {
    /// Process a single raw frame
    /// this is called from inside the thread started by self.start
    fn process(&mut self, frame: RawVideoFrame) -> Result<()>;
    fn thread_setup(&mut self) -> Result<()> {
        Ok(())
    }
    fn thread_teardown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Default impl for all VideoEncoders which use a normal processing thread
impl<T> StartVideoEncoder for T
where
    T: ProcessingThread,
{
    fn start_processing(
        capture: &mut crate::Capture<Self>,
        input: Receiver<RawVideoFrame>,
    ) -> Result<()> {
        let encoder = Arc::clone(
            capture
                .video_encoder
                .as_mut()
                .expect("start_processing should be called after Capture.video_encoder is set"),
        );
        let controls = Arc::clone(&capture.controls);

        let handle = std::thread::spawn(move || -> Result<()> {
            encoder.as_ref().lock().unwrap().thread_setup()?;

            let ret = default_processing_loop(input, controls, Arc::clone(&encoder));

            encoder.as_ref().lock().unwrap().thread_teardown()?;
            ret
        });
        capture.worker_handles.push(handle);
        Ok(())
    }
}

/// Default processing loop function. Handles stop/pause and frame interval changes
pub fn default_processing_loop<V: ProcessingThread>(
    input: Receiver<RawVideoFrame>,
    controls: Arc<CaptureControls>,
    thread_self: Arc<Mutex<V>>,
) -> Result<()> {
    let mut last_timestamp: u64 = 0;
    let mut frame_interval = controls.frame_interval_ns();

    while !controls.is_stopped() {
        if controls.is_paused() {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }
        select! {
            recv(input) -> raw_frame => {
                match raw_frame {
                    Ok(raw_frame) => {
                        let current_time = raw_frame.timestamp as u64;
                        if current_time >= last_timestamp + frame_interval {
                            thread_self.lock().unwrap().process(raw_frame)?;
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
                // Timeout to change fps if needed and check stop/pause flags periodically
                frame_interval = controls.frame_interval_ns();
            }
        }
    }
    Ok(())
}

pub trait PipewireSPA {
    fn get_spa_definition() -> Result<spa::pod::Object>;
}

pub fn create_hw_frame_ctx(device: *mut AVBufferRef) -> Result<*mut AVBufferRef> {
    unsafe {
        let frame = av_hwframe_ctx_alloc(device);

        if frame.is_null() {
            return Err(WaycapError::Init(
                "Could not create hw frame context".to_string(),
            ));
        }

        Ok(frame)
    }
}

pub fn create_hw_device(device_type: ffmpeg_next::ffi::AVHWDeviceType) -> Result<*mut AVBufferRef> {
    unsafe {
        let mut device: *mut AVBufferRef = null_mut();
        let device_path = CString::new("/dev/dri/renderD128").unwrap();
        let ret = av_hwdevice_ctx_create(
            &mut device,
            device_type,
            device_path.as_ptr(),
            null_mut(),
            0,
        );
        if ret < 0 {
            return Err(WaycapError::Init(format!(
                "Failed to create hardware device: Error code {ret:?}",
            )));
        }

        Ok(device)
    }
}
