use ffmpeg_next::{self as ffmpeg, Rational};
use ringbuf::{
    traits::{Producer, Split},
    HeapCons, HeapProd, HeapRb,
};

use crate::types::{
    config::QualityPreset,
    error::Result,
    video_frame::{EncodedVideoFrame, RawVideoFrame},
};

use super::video::{VideoEncoder, GOP_SIZE};

pub struct NvencEncoder {
    encoder: Option<ffmpeg::codec::encoder::Video>,
    width: u32,
    height: u32,
    encoder_name: String,
    quality: QualityPreset,
    encoded_frame_recv: Option<HeapCons<EncodedVideoFrame>>,
    encoded_frame_sender: Option<HeapProd<EncodedVideoFrame>>,
}

impl VideoEncoder for NvencEncoder {
    fn new(width: u32, height: u32, quality: QualityPreset) -> Result<Self>
    where
        Self: Sized,
    {
        let encoder_name = "h264_nvenc";
        let encoder = Self::create_encoder(width, height, encoder_name, &quality)?;
        let video_ring_buffer = HeapRb::<EncodedVideoFrame>::new(120);
        let (video_ring_sender, video_ring_receiver) = video_ring_buffer.split();

        Ok(Self {
            encoder: Some(encoder),
            width,
            height,
            encoder_name: encoder_name.to_string(),
            quality,
            encoded_frame_recv: Some(video_ring_receiver),
            encoded_frame_sender: Some(video_ring_sender),
        })
    }

    fn process(&mut self, frame: &RawVideoFrame) -> Result<()> {
        if let Some(ref mut encoder) = self.encoder {
            let mut src_frame = ffmpeg::util::frame::video::Video::new(
                ffmpeg_next::format::Pixel::BGRA,
                encoder.width(),
                encoder.height(),
            );

            src_frame.set_pts(Some(frame.timestamp));
            src_frame.data_mut(0).copy_from_slice(&frame.data);

            encoder.send_frame(&src_frame).unwrap();

            // Retrieve and handle the encoded packet
            let mut packet = ffmpeg::codec::packet::Packet::empty();
            if encoder.receive_packet(&mut packet).is_ok() {
                if let Some(data) = packet.data() {
                    if let Some(ref mut sender) = self.encoded_frame_sender {
                        if sender
                            .try_push(EncodedVideoFrame {
                                data: data.to_vec(),
                                is_keyframe: packet.is_key(),
                                pts: packet.pts().unwrap_or(0),
                                dts: packet.dts().unwrap_or(0),
                            })
                            .is_err()
                        {
                            log::error!("Could not send encoded packet to the ringbuf");
                        }
                    }
                };
            }
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<()> {
        if let Some(ref mut encoder) = self.encoder {
            // Drain encoder
            encoder.send_eof()?;
            let mut packet = ffmpeg::codec::packet::Packet::empty();
            while encoder.receive_packet(&mut packet).is_ok() {
                if let Some(data) = packet.data() {
                    if let Some(ref mut sender) = self.encoded_frame_sender {
                        if sender
                            .try_push(EncodedVideoFrame {
                                data: data.to_vec(),
                                is_keyframe: packet.is_key(),
                                pts: packet.pts().unwrap_or(0),
                                dts: packet.dts().unwrap_or(0),
                            })
                            .is_err()
                        {
                            log::error!("Could not send encoded packet to the ringbuf");
                        }
                    }
                };
                packet = ffmpeg::codec::packet::Packet::empty();
            }
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.drop_encoder();
        let new_encoder =
            Self::create_encoder(self.width, self.height, &self.encoder_name, &self.quality)?;

        self.encoder = Some(new_encoder);
        Ok(())
    }

    fn get_encoder(&self) -> &Option<ffmpeg::codec::encoder::Video> {
        &self.encoder
    }

    fn drop_encoder(&mut self) {
        self.encoder.take();
    }

    fn take_encoded_recv(&mut self) -> Option<HeapCons<EncodedVideoFrame>> {
        self.encoded_frame_recv.take()
    }
}

impl NvencEncoder {
    fn create_encoder(
        width: u32,
        height: u32,
        encoder: &str,
        quality: &QualityPreset,
    ) -> Result<ffmpeg::codec::encoder::Video> {
        let encoder_codec =
            ffmpeg::codec::encoder::find_by_name(encoder).ok_or(ffmpeg::Error::EncoderNotFound)?;

        let mut encoder_ctx = ffmpeg::codec::context::Context::new_with_codec(encoder_codec)
            .encoder()
            .video()?;

        encoder_ctx.set_width(width);
        encoder_ctx.set_height(height);
        encoder_ctx.set_format(ffmpeg::format::Pixel::BGRA);
        encoder_ctx.set_bit_rate(16_000_000);

        // These should be part of a config file
        encoder_ctx.set_time_base(Rational::new(1, 1_000_000));

        // Needed to insert I-Frames more frequently so we don't lose full seconds
        // when popping frames from the front
        encoder_ctx.set_gop(GOP_SIZE);

        let encoder_params = ffmpeg::codec::Parameters::new();

        let opts = Self::get_encoder_params(quality);

        encoder_ctx.set_parameters(encoder_params)?;
        let encoder = encoder_ctx.open_with(opts)?;

        Ok(encoder)
    }

    fn get_encoder_params(quality: &QualityPreset) -> ffmpeg::Dictionary {
        let mut opts = ffmpeg::Dictionary::new();
        opts.set("vsync", "vfr");
        opts.set("rc", "vbr");
        opts.set("tune", "hq");
        match quality {
            QualityPreset::Low => {
                opts.set("preset", "p2");
                opts.set("cq", "30");
                opts.set("b:v", "20M");
            }
            QualityPreset::Medium => {
                opts.set("preset", "p4");
                opts.set("cq", "25");
                opts.set("b:v", "40M");
            }
            QualityPreset::High => {
                opts.set("preset", "p7");
                opts.set("cq", "20");
                opts.set("b:v", "80M");
            }
            QualityPreset::Ultra => {
                opts.set("preset", "p7");
                opts.set("cq", "15");
                opts.set("b:v", "120M");
            }
        }
        opts
    }
}

impl Drop for NvencEncoder {
    fn drop(&mut self) {
        if let Err(e) = self.drain() {
            log::error!("Error while draining nvenc encoder during drop: {:?}", e);
        }
        self.drop_encoder();
    }
}
