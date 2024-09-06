use std::error::Error;
use opencv::core::CV_32F;
use opencv::objdetect;
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
) -> Result<(), Box<dyn Error>> {
    let mut faces = Vector::new();

    loop {
        cam.read(frame)?;
        if frame.size()?.width == 0 || frame.size()?.height == 0 {
            return Err("Failed to get picture".into());
        }

        let mut gray_frame = Mat::default();
        imgproc::cvt_color(frame, &mut gray_frame, imgproc::COLOR_BGR2GRAY, 0)?;

        face_classifier.detect_multi_scale(
            &gray_frame,
            &mut faces,
            1.1,
            2,
            objdetect::CASCADE_SCALE_IMAGE,
            Size::new(300, 300),
            Size::new(4000, 4000),
        )?;

        if faces.len() > 0 {
            println!("[+] Face detected!");
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
    }

    Ok(())
}
