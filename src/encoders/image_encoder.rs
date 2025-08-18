use crate::VideoEncoder;
use crossbeam::channel::{bounded, Receiver, Sender};

struct ImageEncoder {
    image_sender: Sender<image::RgbaImage>,
    image_receiver: Receiver<image::RgbaImage>,
}

impl VideoEncoder for ImageEncoder {
    type Output = image::RgbaImage;

    fn process(
        &mut self,
        frame: &crate::types::video_frame::RawVideoFrame,
    ) -> crate::types::error::Result<()> {
        // todo
    }

    fn reset(&mut self) -> crate::types::error::Result<()> {
        Ok(())
    }

    fn output(&mut self) -> Option<crossbeam::channel::Receiver<Self::Output>> {
        self.image_receiver.clone()
    }

    fn drop_processor(&mut self) {
        Ok(())
    }

    fn drain(&mut self) -> crate::types::error::Result<()> {
        Ok(())
    }

    fn get_encoder(&self) -> &Option<ffmpeg_next::codec::encoder::Video> {
        None
    }
}
