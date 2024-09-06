import torch
import torchvision.models as models

# Load MobileNetV2 model
mobilenet_v2 = models.mobilenet_v2(pretrained=True)
mobilenet_v2.eval()

# Dummy input for the model (Batch Size 1, 3 color channels, 224x224 image size)
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX format with input name "input"
torch.onnx.export(mobilenet_v2, dummy_input, "../models/mobilenetv2.onnx",
                  opset_version=11, input_names=["input"])
