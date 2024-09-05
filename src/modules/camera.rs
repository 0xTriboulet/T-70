use opencv::{videoio, Result};
use opencv::prelude::VideoCaptureTraitConst;

pub fn init_camera() -> Result<videoio::VideoCapture> {
    let cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        panic!("Unable to open default camera!");
    }
    Ok(cam)
}
