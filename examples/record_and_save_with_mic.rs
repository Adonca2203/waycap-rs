use std::{
    collections::BTreeMap,
    sync::{atomic::AtomicBool, Arc, Mutex},
    time::{Duration, Instant},
};

use waycap_rs::{
    pipeline::builder::CaptureBuilder,
    types::{
        audio_frame::EncodedAudioFrame,
        config::{AudioEncoder, QualityPreset},
        error::Result,
        video_frame::EncodedVideoFrame,
    },
    Capture, DynamicEncoder,
};

fn main() -> Result<()> {
    simple_logging::log_to_stderr(log::LevelFilter::Trace);
    log::info!("Capture and Save Example - System Audio + Mic");
    log::info!("=====================");
    log::info!("This example will capture your screen, system audio, and default mic");
    log::info!("(mixed into a single audio track) for 15 seconds, muting the mic");
    log::info!("for the middle 5 seconds to demonstrate CaptureControls::toggle_mic");
    log::info!("and save a file called example_with_mic.mp4 to the current directory");
    log::info!("");
    log::info!("Press Enter to start...");

    // Wait for user input
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let stop = Arc::new(AtomicBool::new(false));

    let mut capture = CaptureBuilder::new()
        .with_quality_preset(QualityPreset::Medium)
        .with_audio()
        .with_mic()
        .with_audio_encoder(AudioEncoder::Opus)
        .with_cursor_shown()
        .build()?;

    let video_recv = capture.get_video_receiver();

    // Use a BTree Map so it is sorted by DTS
    // needed like this for export time monotonic dts times
    let encoded_video = Arc::new(Mutex::new(BTreeMap::<i64, EncodedVideoFrame>::new()));
    let capture_clone = Arc::clone(&encoded_video);
    let h1stop = Arc::clone(&stop);
    let handle1 = std::thread::spawn(move || {
        while !h1stop.load(std::sync::atomic::Ordering::Acquire) {
            match video_recv.recv_timeout(Duration::from_millis(100)) {
                Ok(encoded_frame) => {
                    capture_clone
                        .lock()
                        .unwrap()
                        .insert(encoded_frame.dts, encoded_frame);
                }
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                    log::info!("Video channel disconnected");
                    break;
                }
            }
        }
    });

    let h2stop = Arc::clone(&stop);
    let encoded_audio = Arc::new(Mutex::new(Vec::<EncodedAudioFrame>::new()));
    let audio_clone = Arc::clone(&encoded_audio);
    let audio_recv = capture.get_audio_receiver().unwrap();
    let handle2 = std::thread::spawn(move || {
        while !h2stop.load(std::sync::atomic::Ordering::Acquire) {
            match audio_recv.recv_timeout(Duration::from_millis(100)) {
                Ok(encoded_frame) => {
                    audio_clone.lock().unwrap().push(encoded_frame);
                }
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                    log::info!("Video channel disconnected");
                    break;
                }
            }
        }
    });

    log::info!("Starting 15-second capture...");
    capture.start()?;

    let start_time = Instant::now();
    let capture_duration = Duration::from_secs(15);
    let controls = capture.controls();
    // One-shot latches so each demo event fires exactly once instead of re-triggering every tick.
    let mut has_muted = false;
    let mut has_unmuted = false;

    while start_time.elapsed() < capture_duration {
        let elapsed = start_time.elapsed();

        if !has_muted && elapsed >= Duration::from_secs(5) {
            log::info!("Muting mic for 5 seconds...");
            controls.toggle_mic();
            has_muted = true;
        }
        if !has_unmuted && elapsed >= Duration::from_secs(10) {
            log::info!("Unmuting mic...");
            controls.toggle_mic();
            has_unmuted = true;
        }

        std::thread::sleep(Duration::from_millis(100));
    }

    log::info!("Capture complete! Stopping...");
    stop.store(true, std::sync::atomic::Ordering::Release);

    let _ = handle1.join();
    let _ = handle2.join();

    log::info!("Writing output to example_with_mic.mp4");

    save_buffer(
        "example_with_mic.mp4",
        &encoded_video.lock().unwrap(),
        &encoded_audio.lock().unwrap(),
        &capture,
    )?;

    Ok(())
}

fn save_buffer(
    filename: &str,
    video_buffer: &BTreeMap<i64, EncodedVideoFrame>,
    audio_buffer: &Vec<EncodedAudioFrame>,
    capture: &Capture<DynamicEncoder>,
) -> Result<()> {
    let mut output = ffmpeg_next::format::output(&filename)?;

    capture.with_video_encoder(|enc| {
        if let Some(encoder) = enc {
            let video_codec = encoder.codec().unwrap();
            let mut video_stream = output.add_stream(video_codec).unwrap();
            video_stream.set_time_base(encoder.time_base());
            video_stream.set_parameters(encoder);
        }
    });

    capture.with_audio_encoder(|enc| {
        if let Some(encoder) = enc {
            let audio_codec = encoder.codec().unwrap();
            let mut audio_stream = output.add_stream(audio_codec).unwrap();
            audio_stream.set_time_base(encoder.time_base());
            audio_stream.set_parameters(encoder);
        }
    });

    output.write_header()?;

    let first_pts = video_buffer
        .values()
        .next()
        .map(|frame| frame.pts)
        .unwrap_or(0);

    // Write video
    for frame in video_buffer.values() {
        let mut packet = ffmpeg_next::codec::packet::Packet::copy(&frame.data);
        packet.set_pts(Some(frame.pts - first_pts));
        packet.set_dts(Some(frame.dts - first_pts));

        // 0 = Video
        // 1 = Audio
        // these should be in the same order we set them above
        packet.set_stream(0);

        packet.write_interleaved(&mut output)?;
    }

    let first_pts = audio_buffer.first().map(|f| f.pts).unwrap_or(0);
    // Write Audio
    for sample in audio_buffer {
        let mut packet = ffmpeg_next::codec::packet::Packet::copy(&sample.data);
        packet.set_pts(Some(sample.pts - first_pts));
        packet.set_dts(Some(sample.pts - first_pts));

        packet.set_stream(1);

        packet.write_interleaved(&mut output)?;
    }

    output.write_trailer()?;

    Ok(())
}
