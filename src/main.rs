use opencv::core::CV_32F;
use crate::imgcodecs::imwrite;
use std::error::Error;
use opencv::core::{FileStorage, FileStorage_Mode, Rect, Scalar, Size, Vector};
use opencv::dnn::{blob_from_image, read_net_from_onnx_buffer};
use opencv::prelude::*;
use opencv::{highgui, imgcodecs, imgproc, objdetect, videoio, Result};
use opencv::objdetect::CascadeClassifier;

const REFERENCE_IMAGE_DATA: &[u8] = include_bytes!("../images/reference_image.png");
const RESNET_MODEL_DATA: &[u8] = include_bytes!("../models/resnet50.onnx");

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
            Size::new(30, 30),
            Size::new(800, 800),
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

        // Save the cropped face image
        imwrite("./detected_face.jpeg", &face_mat, &Vector::new())?;

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
            1.0,
            Size::new(224, 224),
            Scalar::default(),
            true,  // SwapRB - if your image is in BGR format (which OpenCV usually uses), this converts it to RGB
            false, // Crop
            CV_32F,
        )?;

        embeddings.push(extract_embedding(&RESNET_MODEL_DATA, &mut blob)?);
    }

    // Save the image with rectangles
    imwrite("./captured.jpeg", &frame, &Vector::new())?;

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

        // Save the cropped face image from reference image
        imwrite("./reference_face.jpeg", &reference_face_mat, &Vector::new())?;

        // Resize the reference face image to 224x224
        let mut resized_reference_face = Mat::default();
        imgproc::resize(&reference_face_mat, &mut resized_reference_face, Size::new(224, 224), 0.0, 0.0, imgproc::INTER_LINEAR)?;

        // Convert the image to blob format
        let mut blob = blob_from_image(
            &resized_reference_face,
            1.0,
            Size::new(224, 224),
            Scalar::default(),
            true,  // SwapRB - if your image is in BGR format (which OpenCV usually uses), this converts it to RGB
            false, // Crop
            CV_32F,
        )?;
        embeddings.push(extract_embedding(&RESNET_MODEL_DATA, &mut blob)?);
    }

    const WINDOW_1: &str = "capture";
    highgui::named_window_def(WINDOW_1)?;

    const WINDOW_2: &str = "reference";
    highgui::named_window_def(WINDOW_2)?;

    // Show detected face
    highgui::imshow(WINDOW_1, &frame).expect("Failed!");
    highgui::wait_key(0)?;

    // Show reference face
    highgui::imshow(WINDOW_2, &reference_image).expect("Failed!");
    highgui::wait_key(0)?;

    Ok(())
}

fn extract_embedding(model_data: &[u8], input_blob: &mut Mat) -> Result<Vector<f32>> {
    // Convert the &[u8] to Vector<u8>
    let model_data_vector = Vector::from_slice(model_data);

    // Load the ONNX model
    let mut net = read_net_from_onnx_buffer(&model_data_vector)?;

    // Set the input for the network
    net.set_input(input_blob, "input", 1.0, Scalar::default())?;

    // Prepare an output Mat to store the result
    let mut output = Mat::default();

    // Prepare a Vector<String> for output blob names, here we assume the output layer name is not required
    let output_blob_names = Vector::<String>::new();

    // Perform forward pass
    net.forward(&mut output, &output_blob_names)?;

    // Directly access the data from the Mat
    let mut embedding = Vector::<f32>::new();
    if let Ok(data) = output.data_typed::<f32>() {
        data.iter().for_each(|&val| embedding.push(val));
    } else {
        return Err(opencv::Error::new(0, "Failed to access output data"));
    }

    Ok(embedding)
}
