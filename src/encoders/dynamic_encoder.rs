use crossbeam::channel::Receiver;
use ffmpeg_next::codec::encoder;

use crate::{
    encoders::{nvenc_encoder::NvencEncoder, vaapi_encoder::VaapiEncoder},
    types::{
        config::VideoEncoder,
        error::Result,
        video_frame::{EncodedVideoFrame, RawVideoFrame},
    },
    RawProcessor,
};

pub struct DynamicEncoder(Box<dyn RawProcessor<Output = EncodedVideoFrame>>);

impl DynamicEncoder {
    pub(crate) fn new(
        encoder_type: VideoEncoder,
        width: u32,
        height: u32,
        quality_preset: crate::types::config::QualityPreset,
    ) -> crate::types::error::Result<DynamicEncoder> {
        let video_encoder = match encoder_type {
            VideoEncoder::H264Nvenc => {
                DynamicEncoder(Box::new(NvencEncoder::new(width, height, quality_preset)?))
            }
            VideoEncoder::H264Vaapi => {
                DynamicEncoder(Box::new(VaapiEncoder::new(width, height, quality_preset)?))
            }
        };
        Ok(video_encoder)
    }
}

impl RawProcessor for DynamicEncoder {
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
}
