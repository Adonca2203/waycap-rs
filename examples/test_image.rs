use waycap_rs::{types::error::Result, Capture, ImageEncoder, VideoEncoder};

fn main() -> Result<()> {
    simple_logging::log_to_stderr(log::LevelFilter::Debug);
    let mut imgenc = ImageEncoder::default();
    let recv = imgenc.output().unwrap();
    let mut cap = Capture::new_with_encoder(imgenc, false, 30).unwrap();
    println!("Starting capture...");
    cap.start().unwrap();
    loop {
        let img = recv.recv();
    }

    Ok(())
}
