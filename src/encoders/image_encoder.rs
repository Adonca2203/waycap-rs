use crate::VideoEncoder;
use crossbeam::channel::{bounded, Receiver, Sender};
use image::RgbaImage;
pub struct ImageEncoder {
    image_sender: Sender<image::RgbaImage>,
    image_receiver: Receiver<image::RgbaImage>,
}

impl Default for ImageEncoder {
    fn default() -> Self {
        let (image_sender, image_receiver) = crossbeam::channel::bounded(10);
        Self {
            image_sender,
            image_receiver,
        }
    }
}

impl VideoEncoder for ImageEncoder {
    type Output = image::RgbaImage;

    fn process(
        &mut self,
        frame: &crate::types::video_frame::RawVideoFrame,
    ) -> crate::types::error::Result<()> {
        dbg!(frame);
        Ok(())
    }

    fn reset(&mut self) -> crate::types::error::Result<()> {
        Ok(())
    }

    fn output(&mut self) -> Option<crossbeam::channel::Receiver<Self::Output>> {
        Some(self.image_receiver.clone())
    }

    fn drop_processor(&mut self) {}

    fn drain(&mut self) -> crate::types::error::Result<()> {
        Ok(())
    }

    fn get_encoder(&self) -> &Option<ffmpeg_next::codec::encoder::Video> {
        &None
    }
}

// benchmarked
// bgra
// rgba
pub fn bgra_to_rgba_inplace(buf: &mut [f32]) {
    let loops = buf.len() / 4;
    for i in 0..loops {
        buf.swap(i, i + 2);
    }
}
