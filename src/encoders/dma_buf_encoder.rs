use crate::{
    encoders::video::{PipewireSPA, StartVideoEncoder},
    types::{error::WaycapError, video_frame::RawVideoFrame},
    waycap_egl::{EglContext, GpuVendor},
    NvencEncoder, VaapiEncoder, VideoEncoder,
};
use crossbeam::channel::Receiver;

use crate::types::error::Result;

/// "Encoder" which provides the raw DMA-Buf pointers directly.
///
/// Allows for using the image directly on the GPU, which makes it far more performant when, for example, trying to display it to a user.
/// The implementations of [`crate::NvencEncoder`] and [`crate::VaapiEncoder`] show how a [`RawVideoFrame`] can be used.
#[derive(Default)]
pub struct DmaBufEncoder {
    receiver: Option<Receiver<RawVideoFrame>>,
}

impl StartVideoEncoder for DmaBufEncoder {
    fn start_processing(
        capture: &mut crate::Capture<Self>,
        input: Receiver<RawVideoFrame>,
    ) -> Result<()> {
        capture
            .video_encoder
            .as_mut()
            .expect("start_processing should be called after Capture.video_encoder is set")
            .lock()
            .unwrap()
            .receiver = Some(input);
        Ok(())
    }
}
impl VideoEncoder for DmaBufEncoder {
    type Output = RawVideoFrame;

    fn reset(&mut self) -> crate::types::error::Result<()> {
        Ok(())
    }

    fn output(&mut self) -> Option<Receiver<Self::Output>> {
        self.receiver.clone()
    }

    fn drop_processor(&mut self) {}

    fn drain(&mut self) -> crate::types::error::Result<()> {
        Ok(())
    }

    fn get_encoder(&self) -> &Option<ffmpeg_next::codec::encoder::Video> {
        &None
    }
}

impl PipewireSPA for DmaBufEncoder {
    fn get_spa_definition() -> Result<pipewire::spa::pod::Object> {
        let dummy_context = EglContext::new(100, 100)?;
        match dummy_context.get_gpu_vendor() {
            GpuVendor::NVIDIA => NvencEncoder::get_spa_definition(),
            GpuVendor::AMD | GpuVendor::INTEL => VaapiEncoder::get_spa_definition(),
            GpuVendor::UNKNOWN => Err(WaycapError::Init(
                "Unknown/Unimplemented GPU vendor".to_string(),
            )),
        }
    }
}
