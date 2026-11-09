# waycap-rs

A high-level Wayland screen capture library with hardware-accelerated encoding for Linux environments.

## Features

- **Hardware-accelerated video encoding** (VAAPI for Intel/AMD, NVENC for NVIDIA)
- **Audio capture** with Opus encoding, including system audio, microphone, or both mixed into a single track
- **Device selection** by name (e.g. for a front-end GUI to let users pick their mic) via `list_audio_devices()`
- **Copy-Free** video encoding leveraging PipeWire's DMA Buffers
- **Multiple quality presets** for various use cases
- **Cursor visibility control**
- **Simple, ergonomic API** for easy integration

## Requirements

- Linux with Wayland display server
- XDG Desktop Portal
- PipeWire
- VA-API compatible hardware for VAAPI encoding
- CUDA compatible hardware for NVENC encoding

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
# Default: VAAPI only (Intel/AMD)
waycap-rs = "3.1.0"
crossbeam = "0.8.4"

# NVIDIA only
waycap-rs = { version = "3.1.0", default-features = false, features = ["nvidia", "vulkan"] }

# Support both, auto-detect GPU at runtime
waycap-rs = { version = "3.1.0", features = ["nvidia", "vulkan"] }
```

### Cargo Features

| Feature  | Default | Description |
|----------|---------|-------------|
| `vaapi`  | yes     | Intel/AMD hardware encoding via VAAPI |
| `nvidia` | no      | NVIDIA hardware encoding via NVENC/CUDA |
| `vulkan` | no      | Vulkan interop for DMA-BUF (use with `nvidia`) |
| `egl`    | no      | EGL/OpenGL interop alternative to `vulkan` (use with `nvidia`) |

`vulkan` and `egl` are mutually exclusive. `nvidia` requires one of them.

## Example Usage
```rust
use std::{thread, time::Duration};
use waycap_rs::pipeline::builder::CaptureBuilder;
use waycap_rs::types::config::{AudioEncoder, QualityPreset, VideoEncoder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a capture session
    let mut capture = CaptureBuilder::new()
        .with_audio()
        .with_quality_preset(QualityPreset::Medium)
        .with_cursor_shown()
        .with_video_encoder(VideoEncoder::H264Vaapi)
        .with_audio_encoder(AudioEncoder::Opus)
        .build()?;
    
    // Start capturing
    capture.start()?;
    
    // Get receivers for encoded frames
    let video_receiver = capture.get_video_receiver();
    let audio_receiver = capture.get_audio_receiver()?;
    
    // Process frames in separate threads
    let video_thread = thread::spawn(move || {
        while let Ok(frame) = video_receiver.try_recv() {
            // Process video frame (e.g., save to file, stream, etc.)
            println!("Video frame: keyframe={}, size={}", frame.is_keyframe, frame.data.len());
        }
    });
    
    let audio_thread = thread::spawn(move || {
        while let Ok(frame) = audio_receiver.try_recv() {
            // Process audio frame
            println!("Audio frame: size={}", frame.data.len());
        }
    });
    
    // Capture for 10 seconds
    thread::sleep(Duration::from_secs(10));
    
    // Stop capturing
    capture.close()?;
    
    // Wait for threads to finish
    video_thread.join().unwrap();
    audio_thread.join().unwrap();
    
    Ok(())
}
```

## Audio Sources

By default `.with_audio()` captures system audio (the default playback device). You can also
capture a microphone, either alongside system audio (mixed into one track) or by itself, and
target a specific device by name instead of the system default:

```rust
use waycap_rs::pipeline::builder::CaptureBuilder;
use waycap_rs::types::config::AudioSource;

// System audio + default mic, mixed into one track
let mut capture = CaptureBuilder::new()
    .with_audio()
    .with_mic()
    .build()?;

// Toggle mic mute at any point without restarting the stream
let controls = capture.controls();
controls.toggle_mic();
```

To let users pick a specific device (e.g. in a front-end GUI), enumerate what's available and
pass the chosen device's name via `AudioSource::Named`:

```rust
for device in waycap_rs::list_audio_devices()? {
    println!("{:?}: {} ({})", device.kind, device.description, device.name);
}

let mut capture = CaptureBuilder::new()
    .with_mic_source(AudioSource::Named("USB Microphone".into()))
    .build()?;
```

## Primary Use Case: [WayCap](https://github.com/Adonca2203/WayCap)
This library was created primarily to support the development of WayCap -- a low-latency screen recorder targeting Wayland Linux DEs.
waycap-rs originally lived within this application but was broken out to split library and application logic, you can read more about
that project over at its github page

https://github.com/Adonca2203/WayCap

## Contributing
Contributions are always welcome and encouraged, look around for any open issues you want to tackle and 
feel free to open a PR/Issue yourself.

### Build Requirements

If you would like to contribute, the following system dependencies are needed to compile the application:

- Wayland Desktop Environment
- pipewire
- ffmpeg
- pkgconf
- Rust installation. Get it [here](https://www.rust-lang.org/tools/install)

## Installation of Dependencies example: Arch Linux
```bash
sudo pacman -S \
  pipewire \
  ffmpeg \
  wayland \
  wayland-protocols \
  pkgconf
```

After installing the required dependencies you can clone and compile with
```bash
git clone https://github.com/Adonca2203/waycap-rs.git
cd waycap-rs
cargo build
```

To run any of the examples, you can do so with
```bash
# Default (VAAPI — Intel/AMD)
cargo run --example record_and_save

# System audio + mic mixed into one track, with a mute/unmute demo
cargo run --example record_and_save_with_mic

# NVIDIA only
cargo run --example record_and_save --features "nvidia,vulkan"

# Both encoders, auto-detect GPU
cargo run --example record_and_save --features "vaapi,nvidia,vulkan"
```

Please run the examples before making a PR, to test and debug your changes.

### Areas for Improvement:
- Any optimizations for the library's core capture logic.
- Documentation around the public facing APIs.
- Bug Reports via github Issues
- Platform Testing as I am currently limited by my hardware

## Pull Request Guidelines
- **Fork the repository** based off the `main` branch.
- **Write clear and well documented** code with comments where appropriate.
- **Create new unit tests** if applicable.
- **Add new code examples** in `/examples` if applicable.
- **Include references to the issue** you are resolving in the PR.
