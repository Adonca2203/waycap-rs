use crossbeam::channel::{bounded, Receiver, Sender};
use ffmpeg_next::{self as ffmpeg, Rational};
use std::collections::VecDeque;

use crate::types::audio_frame::EncodedAudioFrame;

use super::audio::{boost_with_rms, AudioEncoder};

pub struct OpusEncoder {
    encoder: Option<ffmpeg::codec::encoder::Audio>,
    next_pts: i64,
    leftover_data: VecDeque<f32>,
    encoded_samples_recv: Option<Receiver<EncodedAudioFrame>>,
    encoded_samples_sender: Sender<EncodedAudioFrame>,
    capture_timestamps: VecDeque<i64>,
}

impl OpusEncoder {
    fn create_encoder() -> crate::types::error::Result<ffmpeg::codec::encoder::Audio> {
        let encoder_codec = ffmpeg::codec::encoder::find(ffmpeg_next::codec::Id::OPUS)
            .ok_or(ffmpeg::Error::EncoderNotFound)?;

        let mut encoder_ctx = ffmpeg::codec::context::Context::new_with_codec(encoder_codec)
            .encoder()
            .audio()?;

        encoder_ctx.set_rate(48000);
        encoder_ctx.set_bit_rate(70_000);
        encoder_ctx.set_format(ffmpeg::format::Sample::F32(
            ffmpeg_next::format::sample::Type::Packed,
        ));
        encoder_ctx.set_time_base(Rational::new(1, 48000));
        encoder_ctx.set_frame_rate(Some(Rational::new(1, 48000)));
        encoder_ctx.set_channel_layout(ffmpeg::channel_layout::ChannelLayout::STEREO);

        let mut encoder = encoder_ctx.open()?;

        // Opus frame size is based on n channels so need to update it
        unsafe {
            (*encoder.as_mut_ptr()).frame_size =
                (encoder.frame_size() as i32 * encoder.channels() as i32) as i32;
        }

        Ok(encoder)
    }
}

impl AudioEncoder for OpusEncoder {
    fn new() -> crate::types::error::Result<Self>
    where
        Self: Sized,
    {
        let encoder = Self::create_encoder()?;
        let (frame_tx, frame_rx): (Sender<EncodedAudioFrame>, Receiver<EncodedAudioFrame>) =
            bounded(10);
        Ok(Self {
            encoder: Some(encoder),
            next_pts: 0,
            leftover_data: VecDeque::with_capacity(10),
            encoded_samples_recv: Some(frame_rx),
            encoded_samples_sender: frame_tx,
            capture_timestamps: VecDeque::with_capacity(10),
        })
    }

    fn process(
        &mut self,
        mut raw_frame: crate::types::audio_frame::RawAudioFrame,
    ) -> crate::types::error::Result<()> {
        if let Some(ref mut encoder) = self.encoder {
            let n_channels = encoder.channels() as usize;
            let total_samples = raw_frame.samples.len();

            if !total_samples.is_multiple_of(n_channels) {
                return Err(crate::types::error::WaycapError::FFmpeg(
                    ffmpeg::Error::InvalidData,
                ));
            }

            let frame_size = encoder.frame_size() as usize;

            // Boost the audio so that even if system audio level is low
            // it's still audible in playback
            boost_with_rms(&mut raw_frame.samples)?;
            self.leftover_data.extend(raw_frame.samples);

            // Send chunked frames to encoder
            while self.leftover_data.len() >= frame_size {
                let frame_samples: Vec<f32> = self.leftover_data.drain(..frame_size).collect();
                let mut frame = ffmpeg::frame::Audio::new(
                    encoder.format(),
                    frame_size,
                    encoder.channel_layout(),
                );

                // Capture time in vec
                frame.plane_mut(0).copy_from_slice(&frame_samples);
                frame.set_pts(Some(self.next_pts));
                frame.set_rate(encoder.rate());

                self.capture_timestamps.push_back(raw_frame.timestamp);
                encoder.send_frame(&frame)?;

                // Try and get a frame back from encoder
                let mut packet = ffmpeg::codec::packet::Packet::empty();
                if encoder.receive_packet(&mut packet).is_ok() {
                    if let Some(data) = packet.data() {
                        let pts = packet.pts().unwrap_or(0);
                        match self.encoded_samples_sender.try_send(EncodedAudioFrame {
                            data: data.to_vec(),
                            pts,
                            timestamp: self.capture_timestamps.pop_front().unwrap_or(0),
                        }) {
                            Ok(_) => {}
                            Err(crossbeam::channel::TrySendError::Full(_)) => {
                                log::error!("Could not send encoded audio frame. Receiver is full");
                            }
                            Err(crossbeam::channel::TrySendError::Disconnected(_)) => {
                                log::error!(
                                    "Could not send encoded audio frame. Receiver disconnected"
                                );
                            }
                        }
                    }
                }

                self.next_pts += frame_size as i64;
            }
        }

        Ok(())
    }

    fn get_encoder(&self) -> &Option<ffmpeg_next::codec::encoder::Audio> {
        &self.encoder
    }

    fn drain(&mut self) -> crate::types::error::Result<()> {
        if let Some(ref mut encoder) = self.encoder {
            encoder.send_eof()?;
            let mut packet = ffmpeg::codec::packet::Packet::empty();
            while encoder.receive_packet(&mut packet).is_ok() {} // Discard frames
        }

        Ok(())
    }

    fn drop_encoder(&mut self) {
        self.encoder.take();
    }

    fn reset(&mut self) -> crate::types::error::Result<()> {
        self.drop_encoder();
        self.capture_timestamps.clear();
        self.encoder = Some(Self::create_encoder()?);

        Ok(())
    }

    fn get_encoded_recv(&mut self) -> Option<Receiver<EncodedAudioFrame>> {
        self.encoded_samples_recv.clone()
    }
}
