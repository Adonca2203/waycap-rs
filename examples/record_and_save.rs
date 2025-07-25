use std::{
    collections::BTreeMap,
    sync::{atomic::AtomicBool, Arc, Mutex},
    time::{Duration, Instant},
};

use waycap_rs::{
    pipeline::builder::CaptureBuilder,
    types::{config::QualityPreset, error::Result, video_frame::EncodedVideoFrame},
    Capture,
};

fn main() -> Result<()> {
    simple_logging::log_to_stderr(log::LevelFilter::Debug);
    log::info!("Simple Capture and Save Example");
    log::info!("=====================");
    log::info!("This example will capture your screen for 10 seconds");
    log::info!("and save a file called example2.mp4 to the current directory");
    log::info!("");
    log::info!("Press Enter to start...");

    // Wait for user input
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let stop = Arc::new(AtomicBool::new(false));

    let mut capture = CaptureBuilder::new()
        .with_quality_preset(QualityPreset::Medium)
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

    log::info!("Starting 10-second capture...");
    capture.start()?;

    let start_time = Instant::now();
    let capture_duration = Duration::from_secs(10);

    while start_time.elapsed() < capture_duration {
        std::thread::sleep(Duration::from_millis(100));
    }

    log::info!("Capture complete! Stopping...");
    stop.store(true, std::sync::atomic::Ordering::Release);

    let _ = handle1.join();

    log::info!("Writing output to example2.mp4");

    save_buffer("example2.mp4", &encoded_video.lock().unwrap(), &capture)?;

    Ok(())
}

fn save_buffer(
    filename: &str,
    video_buffer: &BTreeMap<i64, EncodedVideoFrame>,
    capture: &Capture,
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

    output.write_header()?;

    for frame in video_buffer.values() {
        let mut packet = ffmpeg_next::codec::packet::Packet::copy(&frame.data);
        packet.set_pts(Some(frame.pts));
        packet.set_dts(Some(frame.dts));

        // We only have one stream (out video stream)
        packet.set_stream(0);

        packet.write_interleaved(&mut output)?;
    }

    output.write_trailer()?;

    Ok(())
}
