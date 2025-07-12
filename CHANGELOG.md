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
