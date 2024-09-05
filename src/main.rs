use std::env::temp_dir;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use opencv::imgcodecs::imwrite;
use opencv::core::{find_file_def, FileStorage, FileStorage_Mode};
use opencv::prelude::*;
use opencv::{highgui, imgcodecs, imgproc, objdetect, prelude::*, videoio, Result};
use opencv::core::{Rect, Scalar, Size, Vector, CV_32F};
use opencv::objdetect::CascadeClassifier;

fn main() -> Result<(), Box<dyn Error>>  {
    // Include the Haar Cascade XML file in the binary at compile time
    // const FACE_CASCADE_DATA: &[u8] = include_bytes!("../models/haarcascade_frontalface_default.xml");

    let xml = include_str!("../models/haarcascade_frontalface_default.xml");
    let storage = FileStorage::new_def(xml, i32::from(FileStorage_Mode::READ) | i32::from(FileStorage_Mode::MEMORY))?;
    let mut face_classifier = CascadeClassifier::default()?;
    face_classifier.read(&storage.get_first_top_level_node()?)?;

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        panic!("Unable to open default camera!");
    }

    let mut frame = Mat::default();
    // let mut frame_face_mats = Vector::new();

    let mut faces = Vector::new();

    loop {
        loop {
            cam.read(&mut frame)?;
            if frame.size()?.width == 0 || frame.size()?.height == 0 {
                return Err("Failed to get picture".into());
            } else if frame.size()?.width <= 100 || frame.size()?.height <= 100 {
                continue;
            } else {
                break;
            }
        }

        let mut gray_frame = Mat::default();
        imgproc::cvt_color(&frame, &mut gray_frame, imgproc::COLOR_BGR2GRAY, 0)?;

        face_classifier.detect_multi_scale(
            &gray_frame,
            &mut faces,
            1.1,
            2,
            objdetect::CASCADE_SCALE_IMAGE,
            Size::new(30, 30),
            Size::new(180, 180),
        )?;


        if faces.len() > 0 {
            println!("Face detected!");
            break;
        }
    }

    for face in faces {
        imgproc::rectangle(
            &mut frame,
            Rect::from(face),
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            8,
            0,
        )?;
    }

    // Save the image
    imwrite("./captured.jpeg", &frame, &Vector::new())?;

    const WINDOW: &str = "video capture";
    highgui::named_window_def(WINDOW)?;
    highgui::imshow(WINDOW, &frame).expect("Failed!");
    highgui::wait_key(0)?;

    Ok(())
}