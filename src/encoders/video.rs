use std::ffi::CString;
use std::ptr::null_mut;

use crate::types::error::{Result, WaycapError};
use crate::types::video_frame::RawVideoFrame;
use crate::types::{config::QualityPreset, video_frame::EncodedVideoFrame};
use crate::waycap_egl::EglContext;
use ffmpeg_next::ffi::{av_hwdevice_ctx_create, av_hwframe_ctx_alloc, AVBufferRef};
use ffmpeg_next::{self as ffmpeg, Dictionary};
use ringbuf::HeapCons;

pub const GOP_SIZE: u32 = 30;

pub trait VideoEncoder: Send {
    fn new(width: u32, height: u32, quality: QualityPreset) -> Result<Self>
    where
        Self: Sized;
    fn process(&mut self, frame: &RawVideoFrame) -> Result<()>;
    fn drain(&mut self) -> Result<()>;
    fn reset(&mut self) -> Result<()>;
    fn drop_encoder(&mut self);
    fn get_encoder(&self) -> &Option<ffmpeg::codec::encoder::Video>;
    fn take_encoded_recv(&mut self) -> Option<HeapCons<EncodedVideoFrame>>;
    fn process_egl_texture(
        &mut self,
        id: u32,
        capture_time: i64,
    ) -> Result<()>;
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
                "Failed to create hardware device: Error code {}",
                ret
            )));
        }

        Ok(device)
    }
}

pub fn create_hw_device_with_opts(
    device_type: ffmpeg_next::ffi::AVHWDeviceType,
    opts: Dictionary,
) -> Result<*mut AVBufferRef> {
    unsafe {
        let mut device: *mut AVBufferRef = null_mut();
        let ret =
            av_hwdevice_ctx_create(&mut device, device_type, null_mut(), opts.as_mut_ptr(), 0);
        if ret < 0 {
            return Err(WaycapError::Init(format!(
                "Failed to create hardware device: Error code {}",
                ret
            )));
        }

        Ok(device)
    }
}
