# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-07-05

### Added
- **Core Screen Capture**: High-level Wayland screen capture functionality with hardware-accelerated encoding
- **Hardware-Accelerated Video Encoding**: 
  - VAAPI encoder support for Intel and AMD GPUs
  - NVenc encoder support for NVIDIA GPUs
  - Automatic GPU vendor detection for optimal encoder selection
- **Audio Capture**: System audio capture with Opus encoding via PipeWire
- **Copy-Free Video Encoding**: Zero-copy video encoding leveraging PipeWire's DMA buffers
- **Quality Presets**: Multiple built-in quality presets (Low, Medium, High, Ultra) for various use cases
- **Cursor Control**: Configurable cursor visibility in captures
- **Builder Pattern API**: Ergonomic `CaptureBuilder` for easy session configuration
- **Multi-threaded Processing**: Separate video and audio frame receivers for concurrent processing
- **Platform Support**: Full Linux Wayland desktop environment support

### Features
- **Video Encoders**: 
  - `VideoEncoder::Vaapi` - Hardware-accelerated encoding for Intel/AMD (Tested on AMD not Intel)
  - `VideoEncoder::NVenc` - Hardware-accelerated encoding for NVIDIA (Tested)
- **Audio Encoders**: 
  - `AudioEncoder::Opus` - High-quality audio compression
- **Quality Presets**: 
  - `QualityPreset::Low` - Optimized for file size
  - `QualityPreset::Medium` - Balanced quality and performance
  - `QualityPreset::High` - High quality recording
  - `QualityPreset::Ultra` - Maximum quality
- **Capture Options**:
  - Audio capture toggle
  - Cursor visibility control
  - Configurable video and audio encoders
  - Target FPS for recording (Default: 60)

### Dependencies
- **System Requirements**:
  - Linux with Wayland display server
  - XDG Desktop Portal
  - PipeWire
  - VA-API compatible hardware (for VAAPI encoding)
  - NVIDIA drivers with NVenc support (for NVIDIA encoding)

### Documentation
- Comprehensive README with usage examples
- API documentation with builder pattern examples
- Integration guide for screen recording applications

### Notes
- This library was extracted from the [WayCap](https://github.com/Adonca2203/WayCap) project to provide a reusable screen capture solution
- Designed for low-latency screen recording applications
- Optimized for Wayland desktop environments on Linux

## [1.0.1] - 2025-07-11
### Changed
- `finish()` now discards remaining frames in encoder buffers instead of sending them to receivers, preventing channel overflow errors

## [1.0.2] - 2025-07-13
### Changed
- Pipewire logging is now debug for the `core` logs.

## [2.0.0] - 2025-07-14
### Breaking Changes
- Renamed `take_video_receiver` to `get_video_receiver`
- Renamed `take_audio_receiver` to `get_audio_receiver`
- These now return a copy to the channel instead of give ownership allowing multiple consumers to receive the frames

## [2.1.0] - 2025-08-07
### Changed
- Improved native A/V Sync by:
    - Changed how we internally handle timestamps:
        - We no longer use our own `Instant` and timekeep, instead we rely on pipewire's `pw_stream_get_nsec` method to give us an unified timestamps

## [2.1.1] - 2025-08-07
### Changed
- Made time unit public

## [3.0.0] - TBD
### Breaking Changes
- Encoding backends are now opt-in via Cargo feature flags instead of always compiled
  - `vaapi` — Intel/AMD VAAPI encoding (enabled by default)
  - `nvidia` — NVIDIA NVENC encoding via CUDA (requires `vulkan` or `egl`)
  - `vulkan` — Vulkan interop for DMA-BUF → GPU copy (used with `nvidia`)
  - `egl` — EGL/OpenGL interop alternative to `vulkan` (used with `nvidia`)
- At least one of `vaapi` or `nvidia` must be enabled or the build will fail
- `vulkan` and `egl` are mutually exclusive
- `nvidia` requires either `vulkan` or `egl`

### Added
- XDG portal restore token support — on subsequent launches the screen-recording permission prompt is skipped
  - `CaptureBuilder::with_restore_token(token)` — pass a token saved from a previous session
  - `Capture::restore_token` — read the token returned by the portal after a successful start and persist it for next time

## [3.1.0] - 2026-11-08
### Added
- Microphone capture, mixed into the same audio track as system audio
  - `CaptureBuilder::with_mic()` — capture the default microphone alongside system audio
  - `CaptureBuilder::with_mic_source(AudioSource)` — capture a specific mic/input device instead of the default
  - `CaptureBuilder::with_audio_source(AudioSource)` — capture a specific system-audio target instead of the default playback device
  - `AudioSource` enum (`DefaultOutput`, `DefaultInput`, `Named(String)`) for selecting PipeWire devices by name
  - `waycap_rs::list_audio_devices()` — enumerate available playback/capture devices (name + description), e.g. to populate a device picker in a front-end GUI
  - `CaptureControls::toggle_mic()` / `is_mic_muted()` — mute/unmute the mic leg at runtime without tearing down the stream
### Changed
- Replaced the `pactl`-based default-sink lookup with a native PipeWire registry + metadata query
- Every audio capture stream (system audio and mic) now negotiates a consistent 48kHz/stereo format, which is what makes mixing mic + system audio samples safe
### Fixed
- The automatic RMS gain boost (intended for quiet system audio) no longer applies to mic audio, where it was amplifying the mic's noise floor into audible static
