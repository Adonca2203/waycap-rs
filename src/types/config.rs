#[derive(Debug, Clone, Copy)]
pub enum VideoEncoder {
    #[cfg(feature = "nvidia")]
    H264Nvenc,
    #[cfg(feature = "vaapi")]
    H264Vaapi,
}

#[derive(Debug, Clone, Copy)]
pub enum AudioEncoder {
    Opus,
}

/// Selects which PipeWire audio node to capture from.
#[derive(Debug, Clone, Default)]
pub enum AudioSource {
    /// The system's default playback device (i.e. what you hear) - "system audio".
    #[default]
    DefaultOutput,
    /// The system's default microphone / input device.
    DefaultInput,
    /// Matched by substring against `node.name`/`node.description`; see [`crate::list_audio_devices`].
    Named(String),
}

/// Whether an [`AudioDeviceInfo`] is a playback device or a capture device (e.g. a mic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioDeviceKind {
    Output,
    Input,
}

/// A PipeWire audio node discovered by [`crate::list_audio_devices`].
#[derive(Debug, Clone)]
pub struct AudioDeviceInfo {
    /// The stable PipeWire `node.name`; pass to [`AudioSource::Named`] to target this device.
    pub name: String,
    /// The human-readable `node.description`/`node.nick`, suitable for display in a GUI.
    pub description: String,
    pub kind: AudioDeviceKind,
}

#[derive(Debug, Clone, Copy)]
pub enum QualityPreset {
    Low,
    Medium,
    High,
    Ultra,
}
