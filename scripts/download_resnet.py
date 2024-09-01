import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import tf2onnx

# Load the pre-trained ResNet50 model without the top fully connected layers
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Define the output path for the ONNX model
onnx_model_path = '../models/resnet50_model.onnx'

# Convert the Keras model to ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(base_model, input_signature=spec, opset=13)

# Save the model to the ONNX file
with open(onnx_model_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Model saved to {onnx_model_path}")