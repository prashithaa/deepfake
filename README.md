# 🎭 Multimodal Deepfake Detection System
### Image • Video • Audio Detection using Deep Learning

A multi-modal deepfake detection system that analyzes:

- 🖼 Images  
- 🎥 Videos  
- 🔊 Audio  

using ensemble deep learning models including **Xception, CNN-RNN, and LightGBM**.

🏆 93% Accuracy  
📄 Presented at ICIMIA’23 | Accepted at IJISAE’24  

---

## 🚀 Run the Application

```bash
conda activate deepfake
streamlit run app.py
```

This launches a **Streamlit web interface** where users can upload:

- Image files (.jpg)
- Video files (.mp4)
- Audio files (.mp3 / .wav)

---

## 🧠 Model Architecture

### 🖼 Image Detection
- Model: `cnn_xception.h5`
- Architecture: Xception-based CNN
- Feature extraction using deep convolution layers

### 🎥 Video Detection
- Model: `cnn_rnn_video.h5`
- Architecture: CNN + RNN (temporal modeling)
- Frame-level feature extraction + sequence learning

### 🔊 Audio Detection
- Model: `lgb_librosa.joblib`
- Features: MFCC + Mel Spectrogram
- Classifier: LightGBM

---

## 📂 Project Structure

```text
DeepfakeAudioImage/
│
├── app.py
├── cnn_xception.h5
├── cnn_rnn_video.h5
├── lgb_libosa.joblib
├── commandds.txt
├── final_proj_report.pdf
│
├── Sample Media Files
│   ├── real.jpg
│   ├── fake.jpg
│   ├── real.mp4
│   ├── fake.mp4
│   ├── real.wav
│   ├── fake.wav
│   └── other test samples
```

---

## ⚙️ Installation

### 1️⃣ Create Environment

```bash
conda create -n deepfake python=3.9
conda activate deepfake
```

### 2️⃣ Install Required Packages

```bash
pip install -r requirements.txt
```

(If no requirements.txt exists, install manually:)

```bash
pip install streamlit tensorflow keras lightgbm librosa opencv-python numpy pandas scikit-learn matplotlib
```

---

## 📊 Performance

| Metric | Score |
|--------|--------|
| Accuracy | 93% |
| Precision | High |
| Recall | High |
| F1 Score | High |

The system outperforms:
- SVM
- Logistic Regression
- Random Forest
- MobileNet
- Inception
- EfficientNet

---

## 🔍 Detection Pipeline

1. User uploads media  
2. Preprocessing (resize / frame extraction / MFCC extraction)  
3. Feature extraction  
4. Model inference  
5. Multi-modal fusion  
6. Final prediction → **REAL or FAKE**

---

## 📄 Research & Publication

- ICIMIA 2023 – Presented
- IJISAE 2024 – Accepted

Full academic report included in repository.

---

⭐ If you find this project interesting, consider starring the repository.
