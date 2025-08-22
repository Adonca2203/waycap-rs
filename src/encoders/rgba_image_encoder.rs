use crate::{
    encoders::video::{PipewireSPA, ProcessingThread},
    types::video_frame::RawVideoFrame,
    VideoEncoder,
};
use crossbeam::channel::{Receiver, Sender};

use crate::types::error::Result;
use pipewire as pw;

/// "Encoder" which outputs image::RgbaImage
///
/// This is entirely CPU side, and won't ever be as fast as [`NvencEncoder`] or [`VaapiEncoder`].
/// Don't use this to record video!
/// It will likely benefit from compile time optimizations a lot, due to the BGRA to RGBA image conversion.
pub struct RgbaImageEncoder {
    image_sender: Sender<image::RgbaImage>,
    image_receiver: Receiver<image::RgbaImage>,
}

impl Default for RgbaImageEncoder {
    fn default() -> Self {
        let (image_sender, image_receiver) = crossbeam::channel::bounded(10);
        Self {
            image_sender,
            image_receiver,
        }
    }
}

impl ProcessingThread for RgbaImageEncoder {
    fn process(&mut self, frame: RawVideoFrame) -> Result<()> {
        let mut raw = frame.data.clone();
        bgra_to_rgba_inplace(&mut raw);
        let image =
            image::RgbaImage::from_raw(frame.dimensions.width, frame.dimensions.height, raw)
                .unwrap();
        match self.image_sender.try_send(image) {
            Ok(_) => {}
            Err(crossbeam::channel::TrySendError::Full(_)) => {
                log::error!("Could not send encoded video frame. Receiver is full");
            }
            Err(crossbeam::channel::TrySendError::Disconnected(_)) => {
                log::error!("Could not send encoded video frame. Receiver disconnected");
            }
        }
        Ok(())
    }
}

impl VideoEncoder for RgbaImageEncoder {
    type Output = image::RgbaImage;

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

impl PipewireSPA for RgbaImageEncoder {
    fn get_spa_definition() -> Result<pipewire::spa::pod::Object> {
        Ok(pw::spa::pod::object!(
            pw::spa::utils::SpaTypes::ObjectParamFormat,
            pw::spa::param::ParamType::EnumFormat,
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::MediaType,
                Id,
                pw::spa::param::format::MediaType::Video
            ),
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::MediaSubtype,
                Id,
                pw::spa::param::format::MediaSubtype::Raw
            ),
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::VideoFormat,
                Id,
                pw::spa::param::video::VideoFormat::BGRA
            ),
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::VideoSize,
                Choice,
                Range,
                Rectangle,
                pw::spa::utils::Rectangle {
                    width: 2560,
                    height: 1440
                }, // Default
                pw::spa::utils::Rectangle {
                    width: 1,
                    height: 1
                }, // Min
                pw::spa::utils::Rectangle {
                    width: 4096,
                    height: 4096
                } // Max
            ),
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::VideoFramerate,
                Choice,
                Range,
                Fraction,
                pw::spa::utils::Fraction { num: 240, denom: 1 }, // Default
                pw::spa::utils::Fraction { num: 0, denom: 1 },   // Min
                pw::spa::utils::Fraction { num: 244, denom: 1 }  // Max
            ),
        ))
    }
}

/// BGRA to RGBA pixel buffer conversion
///
/// Will likely benefit from compile time optimizations a lot, especially with SIMD instruction sets enabled.
/// `RUSTFLAGS="-C target-cpu=x86-64-v3"` is a relatively safe bet, as according to steam hardware survey ~95% of people have it.
pub fn bgra_to_rgba_inplace(buf: &mut [u8]) {
    // adapted from: Source: https://users.rust-lang.org/t/the-fastest-way-to-copy-a-buffer-bgra-to-rgba/126651/11
    let (chunked, _) = buf.as_chunks_mut::<4>();

    for p in chunked {
        let bgra = u32::from_be_bytes(*p);
        let argb = bgra.swap_bytes();
        let rgba = argb.rotate_left(8);
        *p = rgba.to_be_bytes();
    }
}
