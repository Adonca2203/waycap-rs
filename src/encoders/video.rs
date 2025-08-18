use std::ffi::CString;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::types::error::{Result, WaycapError};
use crate::types::video_frame::RawVideoFrame;
use crate::TIME_UNIT_NS;
use crossbeam::channel::Receiver;
use crossbeam::select;
use ffmpeg::ffi::{av_hwdevice_ctx_create, av_hwframe_ctx_alloc, AVBufferRef};
use ffmpeg_next::{self as ffmpeg};
use std::sync::Mutex;

pub const GOP_SIZE: u32 = 30;

pub trait RawProcessor: Send {
    type Output;
    /// Process a single raw frame
    /// this is called from inside the thread started by self.start
    fn process(&mut self, frame: &RawVideoFrame) -> Result<()>;
    /// setup inside the thread, before the main loop
    /// Cuda needs this, but others might not
    fn thread_setup(&mut self) -> Result<()> {
        Ok(())
    }
    /// teardown inside the thread, before the main loop
    /// Cuda needs this, but others might not
    fn thread_teardown(&mut self) -> Result<()> {
        Ok(())
    }
    fn reset(&mut self) -> Result<()>;
    fn output(&mut self) -> Option<Receiver<Self::Output>>;
    fn drop_processor(&mut self);
    fn drain(&mut self) -> Result<()>;
    fn get_encoder(&self) -> &Option<ffmpeg::codec::encoder::Video>;
}

pub(crate) fn start_video_loop<T>(
    processor: T,
    input: Receiver<RawVideoFrame>,
    stop: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
    target_fps: u64,
) -> (Arc<Mutex<T>>, JoinHandle<Result<()>>)
where
    T: RawProcessor + Sized + 'static,
{
    let returned_self = Arc::new(Mutex::new(processor));
    let thread_self = Arc::clone(&returned_self);

    let handle = std::thread::spawn(move || -> Result<()> {
        let mut last_timestamp: u64 = 0;
        let frame_interval = TIME_UNIT_NS / target_fps;

        thread_self.lock().unwrap().thread_setup()?;

        while !stop.load(Ordering::Acquire) {
            if pause.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            select! {
                recv(input) -> raw_frame => {
                    match raw_frame {
                        Ok(raw_frame) => {
                            let current_time = raw_frame.timestamp as u64;
                            if current_time >= last_timestamp + frame_interval {
                                thread_self.lock().unwrap().process(&raw_frame)?;
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
        thread_self.lock().unwrap().thread_teardown()?;
        Ok(())
    });
    (returned_self, handle)
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
