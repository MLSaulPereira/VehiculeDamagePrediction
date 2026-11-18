# 🚗 Vehicle Damage Detection

A deep learning web application that detects and classifies **vehicle damage** from an uploaded image. The app uses a fine-tuned **ResNet-50** model served through an interactive **Streamlit** interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.63.0-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

Upload a photo of a car (front or rear), and the model predicts one of six damage categories. The dataset consists of 2,300 images spread across the following classes:

| Class ID | Label          |
| -------- | -------------- |
| 0        | Front Breakage |
| 1        | Front Crushed  |
| 2        | Front Normal   |
| 3        | Rear Breakage  |
| 4        | Rear Crushed   |
| 5        | Rear Normal    |

### 📸 Demo Screenshot

![App Screenshot](/app_screenshot.png)
_(Example of the Streamlit interface predicting "Front Breakage" from an uploaded image)_

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

The final trained model weights are stored in:
model/save_model.pth

> ⚠️ **Note:** The `.pth` file must be placed in the `model/` directory before running the app.

---

## 📂 Project Structure

```text
vehicle-damage-detection/
│
├── app.py                  # Streamlit UI
├── model_helper.py         # Model definition + prediction logic
├── model/
│   └── save_model.pth      # Trained ResNet-50 weights
├── notebook/
│   └── training.ipynb      # Jupyter notebook used for training
├── docs/
│   └── demo_screenshot.png # App screenshots for README
├── requirements.txt
└── README.md

```

⚙️ Installation

1. Clone the repository
   bash
   git clone https://github.com/MLSaulPereira/VehiculeDamagePrediction.git
   cd VehiculeDamagePrediction
2. Create a virtual environment (recommended)
   bash
   python -m venv venv
   source venv/bin/activate # Linux / macOS
   venv\Scripts\activate # Windows
3. Install dependencies
   bash
   pip install -r requirements.txt
   requirements.txt:

txt
streamlit==1.63.0
Pillow==12.3.0
torch==2.15.0
torchvision==0.29.2
💡 For GPU acceleration, install the CUDA-enabled version of PyTorch from pytorch.org.

🚀 Running the App
Since the application code is located inside the streamlit-app directory, navigate into it first:

bash
cd streamlit-app
streamlit run app.py
Then open the URL shown in the terminal (usually http://localhost:8501).

How to Use
Launch the app.

Upload a .jpg or .png image of a vehicle.

The app displays the uploaded image and the predicted damage class below it.

Example flow:

text
Upload Image → Preprocess (224×224, normalize) → ResNet-50 → Softmax → Class Label
🛠️ Tech Stack
Component Technology
Frontend Streamlit
Deep Learning PyTorch + Torchvision
Model ResNet-50 (transfer learning)
Image Processing Pillow (PIL)
🐞 Known Limitations
The model is trained on a limited dataset (2,300 images); accuracy may drop on low-quality, poorly lit, or unusual-angle photos.

Only 6 classes are supported (front/rear breakage, crushed, normal).

The uploaded file is saved temporarily as temp_file.jpg — for multi-user deployment, consider using an in-memory buffer (io.BytesIO) instead.

👤 Author
Saul Pereira

GitHub: @MLSaulPereira

📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

🙏 Acknowledgements
PyTorch

Torchvision

Streamlit

Pretrained ResNet-50 weights from ImageNet
