use opencv::{highgui, prelude::*, videoio, Result};

fn main() -> Result<()> {
    let mut cam = videoio::VideoCapture::new(0,videoio::CAP_ANY)?;

    let mut frame = Mat::default();
    cam.read(&mut frame).expect("Failed to get picture!");

    highgui::imshow("Video", &frame)?;
    highgui::wait_key(10000)?;

    Ok(())
}