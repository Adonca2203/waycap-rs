use waycap_rs::{types::error::Result, Capture, ImageEncoder, VideoEncoder};

fn main() -> Result<()> {
    simple_logging::log_to_stderr(log::LevelFilter::Debug);
    let mut cap = Capture::new_with_encoder(ImageEncoder::default(), false, 30).unwrap();
    let recv = cap.get_output();
    cap.start().unwrap();
    loop {
        let img = recv.recv();
        dbg!(img.unwrap().get_pixel(0, 0));
    }
    Ok(())
}
