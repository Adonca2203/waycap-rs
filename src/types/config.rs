#[derive(Debug, Clone, Copy)]
pub enum VideoEncoder {
    H264Nvenc,
    H264Vaapi,
}

#[derive(Debug, Clone, Copy)]
pub enum AudioEncoder {
    Opus,
}

#[derive(Debug, Clone, Copy)]
pub enum QualityPreset {
    Low,
    Medium,
    High,
    Ultra,
}
