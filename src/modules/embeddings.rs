use opencv::core::{Scalar, Vector, Mat, MatTraitConstManual};
use opencv::dnn::{read_net_from_onnx_buffer};
use opencv::Result;
use opencv::prelude::NetTrait;

const MODEL_DATA: &[u8] = include_bytes!("../../models/mobilenetv2.onnx"); // Mobilnetv2: include_bytes!("../../models/mobilenetv2.onnx");  // or Resnet include_bytes!("../../models/resnet50.onnx");

pub fn extract_embedding(input_blob: &mut Mat) -> Result<Vector<f32>> {
    let model_data_vector = Vector::from_slice(MODEL_DATA);

    let mut net = match read_net_from_onnx_buffer(&model_data_vector) {
        Ok(net) => {
            net
        },
        Err(e) => {
            return Err(e);
        }
    };

    net.set_input(input_blob, "input", 1.0, Scalar::default())?;

    let output = net.forward_single_def()?;
    let data = output.data_typed::<f32>()?;

    let embedding: Vector<f32> = data.iter().cloned().collect();

    Ok(embedding)
}
