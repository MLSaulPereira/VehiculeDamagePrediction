# 🚗 Vehicle Damage Detection API

A deep learning API built with **FastAPI** that classifies vehicle damage from an uploaded image. The model is a fine-tuned **ResNet-50** trained with **PyTorch**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Upload a photo of a car (front or rear), and the API predicts one of six damage categories. The model was trained on a dataset of 2,300 images:

| Class ID | Label          |
| -------- | -------------- |
| 0        | Front Breakage |
| 1        | Front Crushed  |
| 2        | Front Normal   |
| 3        | Rear Breakage  |
| 4        | Rear Crushed   |
| 5        | Rear Normal    |

---

## 🧠 Model Architecture & Training Details

- **Base model:** `ResNet-50` (pretrained on ImageNet, `weights='DEFAULT'`)
- **Transfer learning strategy:**
  - All layers frozen **except** `layer4` and the final fully connected layer (`fc`)
  - Custom head: `Dropout(0.2) → Linear(in_features, 6)`
- **Input preprocessing:**
  - Resize to `224 × 224`
  - Convert to tensor
  - Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
- **Inference:** `torch.no_grad()`, `argmax` over 6 class logits

### ⚙️ Training Hyperparameters

- **Epochs:** 10
- **Optimizer:** Adam (learning rate = 0.005)
- **Loss Function:** CrossEntropyLoss
- **Validation Accuracy:** **82.09%**

The trained weights are stored in `save_model.pth`.

---

## 📂 Project Structure

```text
VehiculeDamagePrediction/
│
├── server.py                        # FastAPI application (endpoints)
├── model_helper.py                  # Model definition + prediction logic
├── save_model.pth                   # Trained ResNet-50 weights
├── requirements.txt
├── practice_damage_prediction.ipynb # Jupyter notebook used for training
├── .gitignore
└── README.md
```
