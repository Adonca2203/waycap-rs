use crate::{
    types::{
        config::{AudioEncoder, QualityPreset, VideoEncoder},
        error::{Result, WaycapError},
    },
    Capture,
};

pub struct CaptureBuilder {
    video_encoder: Option<VideoEncoder>,
    audio_encoder: Option<AudioEncoder>,
    quality_preset: Option<QualityPreset>,
    include_cursor: bool,
    include_audio: bool,
    target_fps: u64,
}

impl Default for CaptureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CaptureBuilder {
    pub fn new() -> Self {
        Self {
            video_encoder: None,
            audio_encoder: None,
            quality_preset: None,
            include_cursor: false,
            include_audio: false,
            target_fps: 60,
        }
    }

    pub fn with_video_encoder(mut self, encoder: VideoEncoder) -> Self {
        self.video_encoder = Some(encoder);
        self
    }

    pub fn with_audio_encoder(mut self, encoder: AudioEncoder) -> Self {
        self.audio_encoder = Some(encoder);
        self
    }

    pub fn with_cursor_shown(mut self) -> Self {
        self.include_cursor = true;
        self
    }

    pub fn with_audio(mut self) -> Self {
        self.include_audio = true;
        self
    }

    pub fn with_quality_preset(mut self, quality: QualityPreset) -> Self {
        self.quality_preset = Some(quality);
        self
    }

    pub fn with_target_fps(mut self, fps: u64) -> Self {
        self.target_fps = fps;
        self
    }

    pub fn build(self) -> Result<Capture> {
        let video_encoder = match self.video_encoder {
            Some(enc) => enc,
            None => {
                return Err(WaycapError::Init(
                    "Video encoder was not specified".to_string(),
                ))
            }
        };

        let quality = match self.quality_preset {
            Some(qual) => qual,
            None => QualityPreset::Medium,
        };

        let audio_encoder = if self.include_audio {
            match self.audio_encoder {
                Some(enc) => enc,
                None => {
                    return Err(WaycapError::Init(
                        "Include audio specified but no audio encoder chosen.".to_string(),
                    ))
                }
            }
        } else {
            AudioEncoder::Opus
        };

        Capture::new(
            video_encoder,
            audio_encoder,
            quality,
            self.include_cursor,
            self.include_audio,
            self.target_fps,
        )
    }
}
