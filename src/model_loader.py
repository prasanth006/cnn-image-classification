import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import json
import os

# Figure out the folder that file lives in, so paths work anywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model info (class names, image size, normalization stats)
with open(os.path.join(BASE_DIR, "model_info.json"), "r") as f:
    INFO = json.load(f)

# Always use CPU for deployment (Render's free tier has no GPU)
DEVICE = torch.device("cpu")

# Build the model structure and load our trained weights (done once at startup)
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, INFO["num_classes"])
    weights_path = os.path.join(BASE_DIR, "cifar10_resnet18_final.pth")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    # Free memory: we never train on the server, so disable gradient tracking globally
    for param in model.parameters():
        param.requires_grad = False
    return model
    
# Load the model a single time when this module is imported
MODEL = load_model()

# The image preprocessing pipeline (must match training)
TRANSFORM = transforms.Compose([
    transforms.Resize(INFO["image_size"]),
    transforms.CenterCrop(INFO["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize(INFO["imagenet_mean"], INFO["imagenet_std"])
])

def predict(image_file):
    """Take an uploaded image file, return prediction info including an 'unknown' flag."""
    image = Image.open(image_file).convert("RGB")
    image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = MODEL(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred = probs.max(1)

    confidence_pct = round(confidence.item() * 100, 2)
    predicted_class = INFO["class_names"][pred.item()]

    # Confidence threshold: if the model isn't sure enough, don't force a guess.
    CONFIDENCE_THRESHOLD = 70.0
    is_confident = confidence_pct >= CONFIDENCE_THRESHOLD

    # Build a full ranked list of all class probabilities for the display
    all_probs = [
        {"class": INFO["class_names"][i], "prob": round(probs[0][i].item() * 100, 2)}
        for i in range(INFO["num_classes"])
    ]
    all_probs.sort(key=lambda x: x["prob"], reverse=True)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence_pct,
        "is_confident": is_confident,
        "all_probs": all_probs
    }