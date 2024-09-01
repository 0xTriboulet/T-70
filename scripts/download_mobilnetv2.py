import torch
import torchvision
import onnx

# Load a pre-trained model like MobileNetV2 from torchvision (as an example)
model = torchvision.models.mobilenet_v2(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Dummy input for the model in NCHW format (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
onnx_model_path = "../models/mobilenet_v2.onnx"
torch.onnx.export(
    model,                       # Model to export
    dummy_input,                 # Dummy input to the model
    onnx_model_path,             # Path to save the ONNX model
    opset_version=11,            # ONNX opset version
    input_names=["input"],       # Input names
    output_names=["output"],     # Output names
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"Model saved to {onnx_model_path}")