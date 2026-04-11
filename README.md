# 🎭 DeepFER — Facial Emotion Recognition Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20%7C%20A100%20GPU-yellow?logo=google-colab)

> An end-to-end deep learning system that classifies human emotions from facial images into **7 categories** using a custom CNN architecture trained on the FER-2013 benchmark dataset.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Emotion Classes](#emotion-classes)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [Future Work](#future-work)
- [Author](#author)

---

## Overview

**DeepFER** is a complete computer vision pipeline for real-time facial emotion recognition. The project covers data exploration, statistical hypothesis testing, class imbalance handling, CNN training with hyperparameter tuning, and a Gradio-powered deployment interface.

**Key capabilities:**
- Classifies 48×48 grayscale facial images into 7 emotion categories
- Handles severe class imbalance using class-weighted loss
- Optimized for A100 GPU via mixed-precision training and XLA JIT compilation
- Includes a full Gradio web interface for live predictions

**Real-world applications:**
- Mental health monitoring (detecting distress or depression signals)
- Customer sentiment analysis in retail/service sectors
- Human-computer interaction systems
- Driver fatigue and alertness monitoring

---

## Emotion Classes

| # | Emotion | Description |
|---|---------|-------------|
| 0 | 😠 Angry | Furrowed brows, tense expression |
| 1 | 🤢 Disgust | Wrinkled nose, raised upper lip |
| 2 | 😨 Fear | Wide eyes, raised brows |
| 3 | 😊 Happy | Upturned mouth, cheek raise |
| 4 | 😐 Neutral | Relaxed, no strong expression |
| 5 | 😢 Sad | Drooping corners, downward gaze |
| 6 | 😲 Surprise | Wide eyes and mouth, raised brows |

---

## Dataset

**FER-2013** (Facial Expression Recognition 2013)

| Property | Value |
|----------|-------|
| Total Images | ~35,887 |
| Image Size | 48 × 48 pixels |
| Channels | Grayscale (1 channel) |
| Train Split | ~28,709 images (~80%) |
| Validation Split | ~7,178 images (~20%) |
| Source | Kaggle / `images/train` + `images/validation` directories |

**Class Imbalance:**
- Most frequent: **Happy** (~8,989 images, ~31%)
- Least frequent: **Disgust** (~547 images, ~2%)
- Imbalance ratio: ~16:1

The dataset is pre-split into `train/` and `validation/` directories organized by emotion class.

**Download the dataset:** [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

---

## Project Structure

```
Facial-Emotion-Recognition/
│
├── Facial_Emotion_Recognition_A100_Fixed.ipynb   # Main notebook (full pipeline)
├── checkpoints/                                   # Saved model checkpoints
│   └── deepfer_final_model_best.keras
├── deepfer_final_model.keras                      # Final saved model (Keras format)
├── deepfer_final_model.h5                         # Final saved model (H5 format)
├── label_mapping.json                             # label2idx / idx2label mapping
└── README.md
```

---

## Methodology

### 1. Exploratory Data Analysis
- 15 visualizations covering univariate, bivariate, and multivariate analysis
- Class distribution bar charts, violin plots, pair plots, pixel intensity histograms
- Average face per emotion class using mean pixel composition

### 2. Statistical Hypothesis Testing
Three formal hypothesis tests were conducted:

| # | Test | Hypothesis | Result |
|---|------|------------|--------|
| H1 | Chi-Square Goodness-of-Fit | Class distribution is uniform | **Rejected** — dataset is significantly imbalanced |
| H2 | Kruskal-Wallis H-test | Pixel intensities are equal across classes | **Rejected** — at least one class differs |
| H3 | Two-sample t-test (one-tailed) | Happy and Sad images have equal brightness | **Rejected** — Happy is significantly brighter |

### 3. Data Preprocessing & Feature Engineering
- **Normalization:** Pixel values divided by 255 → [0, 1] range
- **Class Weights:** Balanced weights computed via `sklearn.utils.compute_class_weight`
  - Disgust: ~5.6× weight | Happy: ~0.4× weight
- **No SMOTE:** Synthetic facial generation introduces artifacts; class-weighted loss is preferred
- **GPU-optimized `tf.data` pipeline:** Cache after normalization, prefetch before GPU

### 4. Model Training — 3 Phases

| Phase | Description | Learning Rate | Dropout |
|-------|-------------|---------------|---------|
| Phase 1 | Baseline run | 1e-3 | 0.4 |
| Phase 2 | Hyperparameter tuning (grid search) | {1e-3, 3e-4, 1e-4} | {0.3, 0.5} |
| Phase 3 | Final optimized run (best params) | Best from Phase 2 | Best from Phase 2 |

---

## Model Architecture

A **Deep CNN** with augmentation layers embedded inside the model (GPU-side):

```
Input (48×48×1)
    │
    ├── Augmentation Block (RandomFlip, RandomRotation, RandomZoom, RandomTranslation)
    │
    ├── Conv2D(32)  → BatchNorm → ReLU → MaxPool → Dropout
    ├── Conv2D(64)  → BatchNorm → ReLU → MaxPool → Dropout
    ├── Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout
    ├── Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout
    │
    ├── GlobalAveragePooling2D
    ├── Dense(256) → BatchNorm → ReLU → Dropout
    └── Dense(7, activation='softmax')   ← float32 cast for mixed precision
```

**Training optimizations:**
- Mixed precision (`float16`) → ~2–3× speedup on A100 Tensor Cores
- XLA JIT compilation → fused GPU kernels
- `tf.data` caching to RAM (eliminates disk I/O after epoch 1)
- Callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline CNN | *(see notebook)* | *(see notebook)* | *(see notebook)* | *(see notebook)* |
| Final Optimized CNN | *(see notebook)* | *(see notebook)* | *(see notebook)* | *(see notebook)* |

> Full per-class metrics, confusion matrices, and training curves are available inside the notebook.

**Business interpretation of metrics:**
- **Recall** is the most critical metric for mental health applications — missing Fear or Disgust is a high-cost error
- **F1-Score** is the primary evaluation metric due to class imbalance
- Accuracy alone is misleading on imbalanced 7-class problems

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (A100 recommended; works on T4/V100 too)
- Google Colab (recommended) or local environment

### Install Dependencies

```bash
pip install tensorflow opencv-python-headless scikit-learn \
            matplotlib seaborn pandas numpy gradio
```

### Dataset Setup

Place the FER-2013 dataset in the following structure:

```
Face Emotion Recognition Dataset/
└── images/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── validation/
        ├── angry/
        └── ... (same structure)
```

Update `TRAIN_DIR` and `VAL_DIR` paths in the notebook to match your directory.

---

## Usage

### Run in Google Colab

1. Upload the notebook to Google Colab
2. Set runtime to **GPU (A100 or T4)**
3. Mount Google Drive and update dataset paths
4. Run all cells sequentially

### Load the Saved Model

```python
import tensorflow as tf
import numpy as np
import cv2
import json

# Load model and label mapping
model = tf.keras.models.load_model('deepfer_final_model.keras')

with open('label_mapping.json', 'r') as f:
    mapping = json.load(f)
idx2label = {int(k): v for k, v in mapping['idx2label'].items()}

# Predict on a single image
def predict_emotion(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=[0, -1])   # shape: (1, 48, 48, 1)

    probs = model.predict(img, verbose=0)[0]
    predicted_idx = np.argmax(probs)
    return idx2label[predicted_idx], float(probs[predicted_idx])

emotion, confidence = predict_emotion('your_face_image.jpg')
print(f"Predicted: {emotion} ({confidence:.2%} confidence)")
```

---

## Deployment

A **Gradio web interface** is included in the notebook for interactive demos:

```python
import gradio as gr

# Launches a shareable link in Colab or a local server
demo.launch(share=True)
```

The interface accepts a live webcam feed or uploaded image and returns:
- Predicted emotion label with emoji
- Confidence score bar chart for all 7 classes

---

## Future Work

- [ ] Real-time webcam inference using OpenCV face detection + DeepFER classification
- [ ] Transfer learning with pre-trained backbones (EfficientNetV2, MobileNetV3)
- [ ] Multi-face detection — detect and classify multiple faces per frame
- [ ] Video emotion timeline — emotion tracking across video frames
- [ ] API deployment — FastAPI or Flask REST endpoint with Docker containerization
- [ ] Attention / Grad-CAM visualization — highlight which facial regions drive predictions
- [ ] Data expansion — combine FER-2013 with AffectNet or RAF-DB for better coverage

---

## Author

**Umerulla**
Project Type: Deep Learning / Image Classification | Contribution: Individual

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Acknowledgements

- **FER-2013 Dataset** — Originally compiled by Pierre-Luc Carrier and Aaron Courville
- **TensorFlow / Keras** — Model building and GPU optimization
- **Gradio** — Interactive deployment interface
- **AlmaBetter** — Academic context and project framework
