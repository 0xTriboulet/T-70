import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model.fc = nn.Identity()  # Remove the classification layer to get embeddings
model.eval()  # Set model to evaluation mode

# Save the model to ONNX format
onnx_model_path = "../models/resnet50.onnx"
os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)  # Create the directory if it doesn't exist

# Create a dummy input tensor with the appropriate shape
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model
torch.onnx.export(model, dummy_input, onnx_model_path,
                  export_params=True, opset_version=11,
                  do_constant_folding=True, input_names=['input'],
                  output_names=['output'])

print(f"ResNet50 model saved to {onnx_model_path}")

