use std::error::Error;
use std::fmt;
use std::io;

#[derive(Debug)]
pub enum WaycapError {
    /// Errors from FFmpeg
    FFmpeg(ffmpeg_next::Error),
    /// Errors from PipeWire
    PipeWire(String),
    /// Errors from XDG Portal
    Portal(String),
    /// I/O errors
    Io(io::Error),
    /// Initialization errors
    Init(String),
    /// Configuration errors
    Config(String),
    /// Stream errors
    Stream(String),
    /// Encoding errors
    Encoding(String),
    /// Device errors
    Device(String),
    /// Validation errors
    Validation(String),
    /// Other errors
    Other(String),
}

impl fmt::Display for WaycapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WaycapError::FFmpeg(err) => write!(f, "FFmpeg error: {}", err),
            WaycapError::PipeWire(msg) => write!(f, "PipeWire error: {}", msg),
            WaycapError::Portal(msg) => write!(f, "XDG Portal error: {}", msg),
            WaycapError::Io(err) => write!(f, "I/O error: {}", err),
            WaycapError::Init(msg) => write!(f, "Initialization error: {}", msg),
            WaycapError::Config(msg) => write!(f, "Configuration error: {}", msg),
            WaycapError::Stream(msg) => write!(f, "Stream error: {}", msg),
            WaycapError::Encoding(msg) => write!(f, "Encoding error: {}", msg),
            WaycapError::Device(msg) => write!(f, "Device error: {}", msg),
            WaycapError::Validation(msg) => write!(f, "Validation error: {}", msg),
            WaycapError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl Error for WaycapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WaycapError::FFmpeg(err) => Some(err),
            WaycapError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<ffmpeg_next::Error> for WaycapError {
    fn from(err: ffmpeg_next::Error) -> Self {
        WaycapError::FFmpeg(err)
    }
}

impl From<io::Error> for WaycapError {
    fn from(err: io::Error) -> Self {
        WaycapError::Io(err)
    }
}

impl From<pipewire::Error> for WaycapError {
    fn from(err: pipewire::Error) -> Self {
        WaycapError::PipeWire(err.to_string())
    }
}

impl From<portal_screencast::PortalError> for WaycapError {
    fn from(err: portal_screencast::PortalError) -> Self {
        WaycapError::Portal(err.to_string())
    }
}

impl From<String> for WaycapError {
    fn from(err: String) -> Self {
        WaycapError::Other(err)
    }
}

impl From<&str> for WaycapError {
    fn from(err: &str) -> Self {
        WaycapError::Other(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, WaycapError>;
