#[derive(Debug)]
pub struct EncodedAudioFrame {
    pub data: Vec<u8>,
    pub pts: i64,
    /// Capture timestamp in micro seconds
    pub timestamp: i64,
}

#[derive(Debug)]
pub struct RawAudioFrame {
    pub samples: Vec<f32>,
    /// Capture timestamp in micro seconds
    pub timestamp: i64,
}
