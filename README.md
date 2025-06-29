# 🚮 Cleanliness Detection Model (Indian Railway Coach Cleanliness)

This project uses a fine-tuned MobileNetV2 Convolutional Neural Network (CNN) to classify railway coach images as **clean** or **dirty**, aiming to automate hygiene monitoring for Indian Railways.

---

## 🔍 Objective

To develop a lightweight, accurate image classifier for assessing the cleanliness of railway coach interiors, reducing manual inspection effort and improving sanitation management.

---

## 🧠 Model Architecture

- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Fine-Tuning**: Last 20 layers unfrozen for domain adaptation
- **Classification Head**:
  - `GlobalAveragePooling2D`
  - `Dense(128, relu)`
  - `Dropout(0.3)`
  - `Dense(1, sigmoid)`

✅ Trained end-to-end using binary crossentropy loss.

---

## 🗃️ Dataset

- Two classes: `clean`, `dirty`
- Image size: 224x224
- Balanced with manual curation and augmentation
- Loaded from `dataset/clean` and `dataset/dirty`

---

## 📊 Evaluation

- **Accuracy**: ~92%
- **Metrics Used**: Accuracy, Precision, Recall, F1 Score
- **Tools**: `sklearn.classification_report`, Confusion Matrix

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- scikit-learn
- Streamlit (for deployment)

---

## 🚀 How to Run

1. **Clone the repo**:
   git clone https://github.com/Trushant123/Cleanliness-model.git
   cd Cleanliness-model

2. **install Dependencies**
   pip install -r requirements.txt
3. **launch WebApp**
   streamlit run app.py
 https://cleanliness-model.onrender.com Live Demo