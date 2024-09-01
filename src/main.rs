use opencv::{highgui, imgcodecs, prelude::*, videoio, Result};
use opencv::core::{Rect, Scalar, Size, Vector, CV_32F};

fn main() -> Result<()> {
    let mut cam = videoio::VideoCapture::new(0,videoio::CAP_ANY)?;

    let mut frame = Mat::default();
    cam.read(&mut frame).expect("Failed to get picture!");

    // Debugging
    // highgui::imshow("Video", &frame)?;
    // highgui::wait_key(3000)?;

    // Load mobilenet for face detection
    let mut model = opencv::dnn::read_net_from_onnx("./models/yolov5.onnx")
                                .expect("Failed to load model");

    // Preprocess the frame for model input

    let mut blob = opencv::dnn::blob_from_image(
        &frame,
        1.0,
        Size::new(224, 224),
        Scalar::new(104.0, 177.0, 123.0, 0.0),
        false,
        true,
        CV_32F,
    )?;

    // "input" name set in python script download_yolov5.py
    model.set_input(&mut blob, "input", 1.0, Default::default())?;


    let mut output_blobs: Vector<Mat> = Vector::new();
    let mut output_blob_names: Vector<String> = Vector::new();

    // "output" name set in python script download_yolov5.py
    output_blob_names.push("output");

    model.forward(&mut output_blobs, &output_blob_names)?;

    let output = output_blobs.get(0)?;

    println!("{:#?}", output);
    let detection_size = 7;  // Adjust based on how many elements per detection (e.g., [image_id, label, confidence, x1, y1, x2, y2])
    let num_detections = output.cols() / detection_size;

    for i in 0..num_detections {
        let confidence = *output.at_2d::<f32>(0, i * detection_size + 2)?; // confidence is the third element in each detection

        if confidence > 0.5 {
            // Get bounding box coordinates (assuming normalized coordinates)
            let x1 = (*output.at_2d::<f32>(0, i * detection_size + 3)? * frame.cols() as f32) as i32;
            let y1 = (*output.at_2d::<f32>(0, i * detection_size + 4)? * frame.rows() as f32) as i32;
            let x2 = (*output.at_2d::<f32>(0, i * detection_size + 5)? * frame.cols() as f32) as i32;
            let y2 = (*output.at_2d::<f32>(0, i * detection_size + 6)? * frame.rows() as f32) as i32;

            // Crop the image to just the face
            let face_rect = Rect::new(x1, y1, x2 - x1, y2 - y1);
            let cropped_face = Mat::roi(&frame, face_rect)?;

            // Save or display the cropped face
            imgcodecs::imwrite("cropped_face.jpg", &cropped_face, &Vector::new())?;
        }
    }

    Ok(())
}