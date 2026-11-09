#[cfg(feature = "nvidia")]
use crate::types::{
    error::Result,
    video_frame::{DmaBufPlane, RawVideoFrame},
};

pub const TIME_UNIT_NS: u64 = 1_000_000_000;

/// Rate/channels every audio stream negotiates to, so mixing two streams' samples is safe.
pub(crate) const TARGET_AUDIO_SAMPLE_RATE: u32 = 48_000;
pub(crate) const TARGET_AUDIO_CHANNELS: u32 = 2;

#[cfg(feature = "nvidia")]
pub fn extract_dmabuf_planes(raw_frame: &RawVideoFrame) -> Result<Vec<DmaBufPlane>> {
    match raw_frame.dmabuf_fd {
        Some(fd) => Ok(vec![DmaBufPlane {
            fd,
            offset: raw_frame.offset,
            stride: raw_frame.stride as u32,
        }]),
        None => Err("No DMA-BUF file descriptor in frame".into()),
    }
}
