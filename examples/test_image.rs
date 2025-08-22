/// Basic example which saves the first frame as a PNG and exits
use waycap_rs::{types::error::Result, Capture, RgbaImageEncoder};

fn main() -> Result<()> {
    simple_logging::log_to_stderr(log::LevelFilter::Trace);
    let mut cap = Capture::new_with_encoder(RgbaImageEncoder::default(), false, 30).unwrap();
    let recv = cap.get_output();
    cap.start().unwrap();

    let img = recv.recv().unwrap();
    img.save("./test.png").unwrap();
    Ok(())
}
