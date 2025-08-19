use crossbeam::channel::Receiver;
use ffmpeg_next::codec::encoder;

use crate::{
    encoders::{nvenc_encoder::NvencEncoder, vaapi_encoder::VaapiEncoder, video::PipewireSPA},
    types::{
        config::VideoEncoder as VideoEncoderType,
        error::{Result, WaycapError},
        video_frame::{EncodedVideoFrame, RawVideoFrame},
    },
    waycap_egl::{EglContext, GpuVendor},
    VideoEncoder,
};

pub struct DynamicEncoder(Box<dyn VideoEncoder<Output = EncodedVideoFrame>>);

impl DynamicEncoder {
    pub(crate) fn new(
        encoder_type: VideoEncoderType,
        width: u32,
        height: u32,
        quality_preset: crate::types::config::QualityPreset,
    ) -> crate::types::error::Result<DynamicEncoder> {
        let video_encoder = match encoder_type {
            VideoEncoderType::H264Nvenc => {
                DynamicEncoder(Box::new(NvencEncoder::new(width, height, quality_preset)?))
            }
            VideoEncoderType::H264Vaapi => {
                DynamicEncoder(Box::new(VaapiEncoder::new(width, height, quality_preset)?))
            }
        };
        Ok(video_encoder)
    }
}

impl VideoEncoder for DynamicEncoder {
    type Output = EncodedVideoFrame;

    fn process(&mut self, frame: &RawVideoFrame) -> Result<()> {
        self.0.process(frame)
    }

    fn reset(&mut self) -> Result<()> {
        self.0.reset()
    }

    fn output(&mut self) -> Option<Receiver<Self::Output>> {
        self.0.output()
    }

    fn drop_processor(&mut self) {
        self.0.drop_processor()
    }

    fn drain(&mut self) -> Result<()> {
        self.0.drain()
    }

    fn get_encoder(&self) -> &Option<encoder::Video> {
        self.0.get_encoder()
    }
    fn thread_setup(&mut self) -> Result<()> {
        self.0.thread_setup()
    }
}

impl PipewireSPA for DynamicEncoder {
    fn get_spa_definition() -> Result<pipewire::spa::pod::Object> {
        let dummy_context = EglContext::new(100, 100)?;
        match dummy_context.get_gpu_vendor() {
            GpuVendor::NVIDIA => NvencEncoder::get_spa_definition(),
            GpuVendor::AMD | GpuVendor::INTEL => VaapiEncoder::get_spa_definition(),
            GpuVendor::UNKNOWN => {
                return Err(WaycapError::Init(
                    "Unknown/Unimplemented GPU vendor".to_string(),
                ));
            }
        }
    }
}
