use crate::types::{
    error::Result,
    video_frame::{DmaBufPlane, RawVideoFrame},
};

pub const TIME_UNIT_NS: u64 = 1_000_000_000;

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

pub fn calculate_dimensions(raw_frame: &RawVideoFrame) -> Result<(u32, u32)> {
    // For ARGB8888: stride = width * 4 bytes per pixel
    let width = (raw_frame.stride / 4) as u32;
    let height = raw_frame.size / raw_frame.stride as u32;

    if width == 0 || height == 0 {
        return Err("Invalid frame dimensions".into());
    }

    Ok((width, height))
}
