use opencv::core::CV_32F;
use std::error::Error;
use opencv::core::{FileStorage, FileStorage_Mode, Rect, Scalar, Size, Vector};
use opencv::dnn::{blob_from_image};
use opencv::prelude::*;
use opencv::{highgui, imgcodecs, imgproc, objdetect, videoio, Result};
use opencv::objdetect::CascadeClassifier;


mod modules; // This points to the `modules` directory

// Import functions from modules
use modules::embeddings::extract_embedding;
use modules::similarity::{calculate_cosine_similarity};
const REFERENCE_IMAGE_DATA: &[u8] = include_bytes!("../images/reference_image.png");

fn main() -> Result<(), Box<dyn Error>> {
    let xml = include_str!("../models/haarcascade_frontalface_default.xml");
    let storage = FileStorage::new_def(xml, i32::from(FileStorage_Mode::READ) | i32::from(FileStorage_Mode::MEMORY))?;
    let mut face_classifier = CascadeClassifier::default()?;
    face_classifier.read(&storage.get_first_top_level_node()?)?;

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        panic!("Unable to open default camera!");
    }

    let mut frame = Mat::default();
    let mut faces = Vector::new();

    // Detect faces from webcam
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
            Size::new(300, 300),
            Size::new(2000, 2000),
        )?;

        if faces.len() > 0 {
            println!("Face detected!");
            break;
        }
    }

    let mut embeddings = Vec::new();

    for face in faces {
        let face_rect = Rect::from(face);
        let mut face_mat = Mat::roi(&frame, face_rect)?;

        imgproc::rectangle(
            &mut frame.clone(),
            face_rect,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            8,
            0,
        )?;

        // Resize the face image to 224x224
        let mut resized_face = Mat::default();
        imgproc::resize(&face_mat, &mut resized_face, Size::new(224, 224), 0.0, 0.0, imgproc::INTER_LINEAR)?;

        // Convert the image to blob format
        let mut blob = blob_from_image(
            &resized_face,
            1.0 / 255.0, // Ensure the same scaling factor
            Size::new(224, 224),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            true,  // SwapRB - if your image is in BGR format (which OpenCV usually uses), this converts it to RGB
            false, // Crop
            CV_32F,
        )?;

        embeddings.push(extract_embedding(&mut blob)?);
    }

    // Process the embedded reference image
    let reference_image = imgcodecs::imdecode(&Vector::from_slice(REFERENCE_IMAGE_DATA), imgcodecs::IMREAD_COLOR)?;

    let mut gray_reference_image = Mat::default();
    imgproc::cvt_color(&reference_image, &mut gray_reference_image, imgproc::COLOR_BGR2GRAY, 0)?;

    let mut reference_faces = Vector::new();
    face_classifier.detect_multi_scale(
        &gray_reference_image,
        &mut reference_faces,
        1.1,
        2,
        objdetect::CASCADE_SCALE_IMAGE,
        Size::new(30, 30),
        Size::new(180, 180),
    )?;

    for reference_face in reference_faces {
        let reference_face_rect = Rect::from(reference_face);
        let mut reference_face_mat = Mat::roi(&reference_image, reference_face_rect)?;

        // Resize the reference face image to 224x224
        let mut resized_reference_face = Mat::default();
        imgproc::resize(&reference_face_mat, &mut resized_reference_face, Size::new(224, 224), 0.0, 0.0, imgproc::INTER_LINEAR)?;

        // Convert the image to blob format
        let mut blob = blob_from_image(
            &resized_reference_face,
            1.0 / 255.0, // Ensure the same scaling factor
            Size::new(224, 224),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            true,  // SwapRB - if your image is in BGR format (which OpenCV usually uses), this converts it to RGB
            false, // Crop
            CV_32F,
        )?;
        embeddings.push(extract_embedding(&mut blob)?);
    }

    const WINDOW_1: &str = "capture";
    highgui::named_window_def(WINDOW_1)?;

    const WINDOW_2: &str = "reference";
    highgui::named_window_def(WINDOW_2)?;

    // Show detected face
    highgui::imshow(WINDOW_1, &frame).expect("Failed!");

    // Show reference face
    highgui::imshow(WINDOW_2, &reference_image).expect("Failed!");
    highgui::wait_key(0)?;


    // Assuming you have two embeddings to compare
    if embeddings.len() >= 2 {
        println!("Number of embeddings: {:#?}", embeddings.len());
        let similarity = calculate_cosine_similarity(&embeddings[1], &embeddings[0]);
        println!("Cosine similarity between embeddings: {}", similarity);
    } else {
        println!("Not enough embeddings to compare.");
    }

    Ok(())
}
