import torch
from ultralytics import YOLO

# Load YOLOv8 model pre-trained for object detection
model = YOLO('yolov8s.pt')  # 'yolov8s.pt' is a small, fast version of YOLOv8; adjust to another variant if needed

# Set the model to evaluation mode
model.eval()

# Dummy input for the model in NCHW format (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 640, 640)  # YOLOv8 typically uses 640x640 input size

# Export the model to ONNX
onnx_model_path = "../models/yolov8.onnx"
torch.onnx.export(
    model,                       # Model to export
    dummy_input,                 # Dummy input to the model
    onnx_model_path,             # Path to save the ONNX model
    opset_version=11,            # ONNX opset version
    input_names=["images"],      # Input names
    output_names=["output"],     # Output names
    dynamic_axes={"images": {0: "batch_size"}, "output": {0: "batch_size"}}  # Dynamic axes
)

print(f"YOLOv8 face detection model saved to {onnx_model_path}")
