# waycap-rs

A high-level Wayland screen capture library with hardware-accelerated encoding for Linux environments.

## Features

- **Hardware-accelerated video encoding** (Using VAAPI or NVENC)
- **Audio capture** with Opus encoding
- **Copy-Free** video encoding leveraging pipewire's DMA Buffers
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
waycap-rs = "1.0.0"
crossbeam = "0.8.4"
```

## Example Usage
```rust
use waycap_rs::{CaptureBuilder, QualityPreset, VideoEncoder, AudioEncoder};
use std::{thread, time::Duration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a capture session
    let mut capture = CaptureBuilder::new()
        .with_audio()
        .with_quality_preset(QualityPreset::Medium)
        .with_cursor_shown()
        .with_video_encoder(VideoEncoder::Vaapi)
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
cargo run --example example_name
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
