# CIFAR-10 Image Classifier — CNN & Transfer Learning (PyTorch)

An end-to-end **machine learning** project that classifies images into 10 object categories using a Convolutional Neural Network built from scratch, then improved with transfer learning. Includes a live web app where you can upload an image and get a prediction with confidence scores.

**🔗 Live Demo:** [cifar10-image-classifier-hpgq.onrender.com](https://cifar10-image-classifier-hpgq.onrender.com)
---

## Headline Result

| Model | Approach | Test Accuracy |
|-------|----------|---------------|
| Custom CNN | Built from scratch (3 conv blocks) | ~78% |
| ResNet18 | Transfer learning (fine-tuned) | **91.71%** |

Transfer learning delivered a **+13.7 percentage point** improvement over the hand-built model — demonstrating both an understanding of CNN fundamentals and the ability to leverage pretrained models effectively.

---

## What It Does

Upload any image and the model predicts which of 10 CIFAR-10 classes it belongs to — **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck** — along with a confidence score and a full probability breakdown across all classes.

The web app includes a **confidence threshold (70%)**: if the model isn't sure enough, it returns "Unknown object" rather than forcing an incorrect guess. This addresses a real limitation of image classifiers trained on a fixed class set (the out-of-distribution problem), which is documented transparently in the app itself.

---

## Tech Stack

- **Deep Learning:** PyTorch, TorchVision
- **Model:** ResNet18 (transfer learning) + a custom CNN baseline
- **Web App:** Flask, Gunicorn
- **Deployment:** Render (CPU-only inference)
- **Environment:** Python, Jupyter Notebook

---

## The ML Pipeline

1. **Data loading & exploration** — CIFAR-10 (60,000 images, 32×32, 10 balanced classes)
2. **Preprocessing** — normalization with per-channel statistics; a train/validation/test split
3. **Data augmentation** — random flips and crops (applied to training data only)
4. **Custom CNN** — 3 convolutional blocks with batch normalization, max pooling, and dropout (~620K parameters)
5. **Training** — cross-entropy loss, Adam optimizer, with validation-based model checkpointing
6. **Evaluation** — learning curves, confusion matrix, and a per-class classification report
7. **Transfer learning** — a pretrained ResNet18 with its final block fine-tuned for CIFAR-10
8. **Deployment** — the best model saved and served through a Flask web app

---

## Key Insights

- **Cat vs. dog confusion:** The custom CNN's weakest class was cat (44% recall), with most misclassified cats predicted as dogs — an intuitive confusion at 32×32 resolution, clearly visible in the confusion matrix.
- **Balanced dataset:** All 10 classes have equal representation (5,000 training images each), so accuracy is a trustworthy metric without class-weighting.
- **Honest model limits:** The classifier only knows its 10 classes. Images outside these (people, text, screenshots) may be misclassified, so a confidence threshold and an on-page notice manage user expectations.

---

## Results & Figures

Figures generated during the project (in `outputs/figures/`):

- `cifar10_samples.png` — sample images from each class
- `augmentation_examples.png` — data augmentation in action
- `learning_curves.png` — training/validation loss and accuracy over epochs
- `confusion_matrix.png` — per-class prediction breakdown

---

## Run It Locally

```bash
# Clone the repository
git clone https://github.com/your-username/cnn-image-classification.git
cd cnn-image-classification

# Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate      # Windows (Git Bash)
# source venv/bin/activate        # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run the web app
cd src
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Project Structure

```
cnn-image-classification/
├── data/                 # CIFAR-10 dataset (auto-downloaded, git-ignored)
├── notebooks/            # Jupyter notebook with the full ML pipeline
├── models/               # Saved model weights (.pth) and model_info.json
├── outputs/figures/      # Generated plots and figures
├── src/
│   ├── app.py            # Flask web app
│   ├── model_loader.py   # Model loading + prediction logic
│   ├── templates/        # HTML front-end
│   └── static/           # CSS styling
├── requirements.txt      # Python dependencies
└── render.yaml           # Render deployment config
```

---

## About

Built as part of a machine learning portfolio focused on computer vision and deep learning. Demonstrates the full lifecycle: data preparation, model building, training, evaluation, and deployment of a working web application.