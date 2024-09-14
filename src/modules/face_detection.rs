use alloc::boxed::Box;
use alloc::vec::Vec;
use opencv::core::CV_32F;
use opencv::{highgui, objdetect};
use opencv::{core::{Rect, Scalar, Size, Vector, Mat}, dnn::blob_from_image, imgproc, objdetect::CascadeClassifier, videoio, Result};
use opencv::objdetect::CascadeClassifierTrait;
use crate::modules::embeddings::extract_embedding;
use opencv::prelude::VideoCaptureTrait;
use opencv::prelude::MatTraitConst;

pub fn detect_faces(
    cam: &mut videoio::VideoCapture,
    face_classifier: &mut CascadeClassifier,
    frame: &mut Mat,
    embeddings: &mut Vec<Vector<f32>>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut faces = Vector::new();

    loop {
        cam.read(frame)?;
        if frame.size()?.width == 0 || frame.size()?.height == 0 {
            return Err("Failed to get picture".into());
        }


        face_classifier.detect_multi_scale(
            frame,
            &mut faces,
            1.1,
            2,
            objdetect::CASCADE_SCALE_IMAGE,
            Size::new(30, 30),
            Size::new(300, 300),
        )?;

        if faces.len() > 0 {
            break;
        }
    }

    // Process detected faces into embeddings
    for face in faces {
        let face_rect = Rect::from(face);
        let face_mat = Mat::roi(frame, face_rect)?;

        let mut resized_face = Mat::default();
        imgproc::resize(&face_mat, &mut resized_face, Size::new(224, 224), 0.0, 0.0, imgproc::INTER_LINEAR)?;

        let mut blob = blob_from_image(
            &resized_face,
            1.0 / 255.0,
            Size::new(224, 224),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            true,
            false,
            CV_32F,
        )?;

        embeddings.push(extract_embedding(&mut blob)?);

        // Show the detected face in a dedicated window
        highgui::imshow("Detected Face", &resized_face)?;
        highgui::wait_key(5000)?; // Wait for 5000 milliseconds (5 seconds)
    }

    Ok(())
}

pub fn process_reference_image(
    face_classifier: &mut CascadeClassifier,
    embeddings: &mut Vec<Vector<f32>>,
) -> Result<()> {
    use opencv::{imgcodecs, core::Vector, imgproc};

    const REFERENCE_IMAGE_DATA: &[u8] = include_bytes!("../../images/reference_image.png");
    let reference_image = imgcodecs::imdecode(&Vector::from_slice(REFERENCE_IMAGE_DATA), imgcodecs::IMREAD_COLOR)?;

    let mut reference_faces = Vector::new();
    face_classifier.detect_multi_scale(
        &reference_image,
        &mut reference_faces,
        1.1,
        2,
        objdetect::CASCADE_SCALE_IMAGE,
        Size::new(100, 100),
        Size::new(1000, 1000),
    )?;

    for reference_face in reference_faces {
        let reference_face_rect = Rect::from(reference_face);
        let reference_face_mat = Mat::roi(&reference_image, reference_face_rect)?;

        let mut resized_reference_face = Mat::default();
        imgproc::resize(&reference_face_mat, &mut resized_reference_face, Size::new(224, 224), 0.0, 0.0, imgproc::INTER_LINEAR)?;

        let mut blob = blob_from_image(
            &resized_reference_face,
            1.0 / 255.0,
            Size::new(224, 224),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            true,
            false,
            CV_32F,
        )?;

        embeddings.push(extract_embedding(&mut blob)?);

        // Show the reference face in its own window
        highgui::imshow("Reference Face", &resized_reference_face)?;
        highgui::wait_key(5000)?; // Wait for 5000 milliseconds (5 seconds)
    }

    Ok(())
}
