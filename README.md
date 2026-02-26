# 🧠 AI-Based Child Health Evaluation
## Deep Learning Model for Malnutrition Detection using Custom CNN

---

## 📌 Project Overview

This project presents a **Custom Convolutional Neural Network (CNN)** designed to classify children as:

- 🟢 Nourished (Healthy)
- 🔴 Malnourished (Unhealthy)

The model automatically predicts nutritional status using facial/body images, enabling scalable and early malnutrition screening.

This system aims to assist healthcare professionals in early identification and intervention.

---

## 🎯 Problem Statement

Traditional malnutrition screening methods:

- Are manual and resource-intensive
- Require trained healthcare professionals
- Are difficult in rural/underserved areas

This project proposes an AI-driven image-based classification system for scalable screening.

---

## 📊 Dataset Details

- Total Training Samples: 1657
- Validation Samples: 503
- Classes:
  - Nourished
  - Malnourished

### Class Distribution:

| Class | Training | Validation |
|--------|----------|------------|
| Nourished | 504 | 126 |
| Malnourished | 1153 | 377 |

Images resized to: **224 x 224 RGB**

---

## 🏗 Custom CNN Architecture

### Architecture Highlights:

- Input: 3 x 224 x 224
- 4 Convolutional Blocks:
  - Conv2D + BatchNorm + ReLU
  - MaxPooling
- Filters increased progressively:
  - 32 → 64 → 128 → 256
- Adaptive Average Pooling (4x4)
- Fully Connected Layers:
  - 512 → 256 → 128 → 2
- Dropout: 0.5
- Loss Function: CrossEntropyLoss
- Optimizer: Adam

---

## 🔬 Experiments Conducted

Multiple experiments were conducted to optimize performance:

<img width="1280" height="720" alt="Slide1" src="https://github.com/user-attachments/assets/431a6d74-bec7-43a8-a46c-67b383cc6cc0" />

| Experiment | Activation | Accuracy |
|------------|------------|----------|
| 50 Epochs | ReLU | 89.66% |
| 100 Epochs | ReLU | 90.66% |
| + Weight Decay | ReLU | **93.44%** |
| Swish | 89.86% |
| LeakyReLU | 92.84% |
| Tanh | 90.46% |
| HardTanh | 88.87% |

---

## 🏆 Best Model Performance

**Experiment 3 (ReLU + Weight Decay + LR Scheduling)**

- Accuracy: **93.44%**
- F1 Score:
  - Healthy: 0.87
  - Malnourished: 0.96
- Precision:
  - Healthy: 0.89
  - Malnourished: 0.95
- Recall:
  - Healthy: 0.85
  - Malnourished: 0.96

---

## 📈 Training Observations

- Stable convergence after 100 epochs
- Learning rate scheduling improved stability
- Weight decay reduced overfitting
- Malnourished class predicted with higher confidence

---

## 🛠 Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn

---

## ⚖ Ethical Considerations

- Image-based health prediction involves sensitive data
- Requires proper consent and privacy compliance
- Model should assist healthcare experts, not replace diagnosis

---

## 🔮 Future Improvements

- Transfer Learning (ResNet, EfficientNet)
- Grad-CAM explainability
- Larger dataset expansion
- Multi-class malnutrition severity detection
- Mobile deployment

---

## 👨‍💻 Author

Kushal G  
M.Tech – Data Science  
