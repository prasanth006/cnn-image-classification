import torch
import torch.nn as nn
from torchvision import models
import json
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cifar10_resnet18_final.pth")
INFO_PATH = os.path.join(BASE_DIR, "models", "model_info.json")
ONNX_OUTPUT = os.path.join(BASE_DIR, "models", "cifar10_resnet18.onnx")

# Load model info to get the number of classes and image size
with open(INFO_PATH, "r") as f:
    info = json.load(f)

# Rebuild the same ResNet18 structure we trained
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, info["num_classes"])

# Load our trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Create a dummy input with the correct shape: (batch=1, channels=3, H, W)
image_size = info["image_size"]
dummy_input = torch.randn(1, 3, image_size, image_size)

# Export to ONNX
torch.onnx.export(
    model,                          # the trained model
    dummy_input,                    # an example input to trace the shape
    ONNX_OUTPUT,                    # where to save the .onnx file
    export_params=True,             # store the trained weights inside the file
    opset_version=11,               # ONNX version (11 is stable and widely supported)
    do_constant_folding=True,       # optimization that simplifies the model
    input_names=["input"],          # a name for the input
    output_names=["output"],        # a name for the output
    dynamic_axes={                  # allow variable batch size
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("✅ Model exported to ONNX successfully!")
print("Saved at:", ONNX_OUTPUT)