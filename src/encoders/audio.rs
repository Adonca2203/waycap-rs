use crossbeam::channel::Receiver;

use crate::types::{
    audio_frame::{EncodedAudioFrame, RawAudioFrame},
    error::Result,
};

const MIN_RMS: f32 = 0.01;

pub trait AudioEncoder: Send {
    fn new() -> Result<Self>
    where
        Self: Sized;
    fn process(&mut self, raw_frame: RawAudioFrame) -> Result<()>;
    fn drain(&mut self) -> Result<()>;
    fn reset(&mut self) -> Result<()>;
    fn get_encoder(&self) -> &Option<ffmpeg_next::codec::encoder::Audio>;
    fn get_encoded_recv(&mut self) -> Option<Receiver<EncodedAudioFrame>>;
    fn drop_encoder(&mut self);
}

pub fn boost_with_rms(samples: &mut [f32]) -> Result<()> {
    let sum_sqrs = samples.iter().map(|&s| s * s).sum::<f32>();
    let rms = (sum_sqrs / samples.len() as f32).sqrt();

    let gain = if rms > 0.0 && rms < MIN_RMS {
        MIN_RMS / rms
    } else {
        1.0
    };

    let gain = gain.min(5.0);
    for sample in samples.iter_mut() {
        *sample *= gain;
    }
    Ok(())
}
