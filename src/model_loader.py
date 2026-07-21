import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import os

# Figure out the folder this file lives in, so paths work anywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model info (class names, image size, normalization stats)
with open(os.path.join(BASE_DIR, "model_info.json"), "r") as f:
    INFO = json.load(f)

# Load the ONNX model once at startup using ONNX Runtime (lightweight)
ONNX_PATH = os.path.join(BASE_DIR, "cifar10_resnet18.onnx")
SESSION = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# Get the input name the model expects (we named it "input" during export)
INPUT_NAME = SESSION.get_inputs()[0].name


def preprocess(image_file):
    """Open an image and turn it into the numpy array the model expects."""
    image = Image.open(image_file).convert("RGB")
    size = INFO["image_size"]

    # Resize so the shorter side is `size`, then center-crop to size x size
    w, h = image.size
    if w < h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_w, new_h = int(w * size / h), size
    image = image.resize((new_w, new_h))

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    image = image.crop((left, top, left + size, top + size))

    # Convert to a numpy array and scale pixels to 0-1
    arr = np.asarray(image).astype("float32") / 255.0

    # Normalize with the same ImageNet stats used in training
    mean = np.array(INFO["imagenet_mean"], dtype="float32")
    std = np.array(INFO["imagenet_std"], dtype="float32")
    arr = (arr - mean) / std

    # Rearrange from (H, W, C) to (C, H, W), then add a batch dimension -> (1, C, H, W)
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def softmax(x):
    """Convert raw model scores into probabilities that sum to 1."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict(image_file):
    """Take an uploaded image file, return prediction info including an 'unknown' flag."""
    input_array = preprocess(image_file)

    # Run the model with ONNX Runtime
    outputs = SESSION.run(None, {INPUT_NAME: input_array})
    logits = outputs[0][0]           # raw scores for the 10 classes

    probs = softmax(logits)
    pred_index = int(np.argmax(probs))
    confidence_pct = round(float(probs[pred_index]) * 100, 2)
    predicted_class = INFO["class_names"][pred_index]

    # Confidence threshold: if the model isn't sure enough, don't force a guess.
    CONFIDENCE_THRESHOLD = 70.0
    is_confident = confidence_pct >= CONFIDENCE_THRESHOLD

    # Build a full ranked list of all class probabilities for the display
    all_probs = [
        {"class": INFO["class_names"][i], "prob": round(float(probs[i]) * 100, 2)}
        for i in range(INFO["num_classes"])
    ]
    all_probs.sort(key=lambda x: x["prob"], reverse=True)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence_pct,
        "is_confident": is_confident,
        "all_probs": all_probs
    }