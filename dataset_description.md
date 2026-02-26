# 📊 Dataset Description
## AI-Based Child Malnutrition Detection

---

## 📌 Overview

This dataset is designed for binary image classification to detect:

- 🟢 Nourished (Healthy)
- 🔴 Malnourished (Unhealthy)

The dataset contains labeled child images used to train a Convolutional Neural Network (CNN) for automated malnutrition detection.

---

## 🗂 Dataset Structure

The dataset follows this folder structure:

dataset/
│
├── train/
│   ├── Nourished/
│   └── Malnourished/
│
├── validation/
│   ├── Nourished/
│   └── Malnourished/


Each subfolder contains RGB images categorized based on nutritional status.

## 📊 Class Distribution

| Class         | Training Samples | Validation Samples |
|--------------|------------------|--------------------|
| Nourished     | 504              | 126                |
| Malnourished  | 1153             | 377                |
| **Total**     | 1657             | 503                |

### ⚠ Class Imbalance

The dataset is imbalanced:

- Malnourished samples are significantly higher than Nourished samples.
- This may bias the model toward the majority class.

To address this:
- Weight decay regularization was applied.
- F1-score and recall were prioritized during evaluation.

---

## 🖼 Image Properties

- Image Type: RGB
- Input Size: 224 x 224 pixels
- Channels: 3
- Format: JPG/PNG (as applicable)

---

## 🔄 Preprocessing Steps

The following transformations were applied:

1. Resize to 224 x 224
2. Convert images to PyTorch tensors
3. Normalize using:
   - Mean: [0.485, 0.456, 0.406]
   - Standard Deviation: [0.229, 0.224, 0.225]

These normalization values are commonly used for ImageNet-style training.

---

## 📈 Data Augmentation

Basic preprocessing was applied:

- Resizing
- Normalization

Future improvements may include:
- Random Horizontal Flip
- Random Rotation
- Color Jitter
- Random Crop

---

## 🔍 Dataset Challenges

This dataset presents multiple challenges:

- Variability in lighting conditions
- Differences in age and posture
- Background noise
- Class imbalance
- Health-related image sensitivity

These factors increase classification complexity.

---

## ⚖ Ethical Considerations

This dataset contains sensitive health-related images.

Important:

- Use strictly for research purposes.
- Ensure privacy protection.
- Do not redistribute without proper authorization.
- Comply with data protection and healthcare regulations.

This AI system is intended to assist healthcare professionals, not replace medical diagnosis.

---

## 🔮 Future Improvements

- Expand dataset size
- Improve class balance
- Add severity-level annotations
- Include demographic diversity
- Integrate explainable AI methods

---
