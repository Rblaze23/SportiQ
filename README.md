# SportiQ
## 🧠 AI-Powered Emotion-Aware Pose Feedback System

This project combines **Computer Vision**, **NLP**, and **Emotion Detection** to create an intelligent system that analyzes human movement and provides **personalized feedback**. It uses pose estimation to understand body movements, NLP to interpret advice, and emotion detection to adapt the feedback tone, delivering a more natural, human-like experience.

## 🔍 Project Overview

The system:
- Extracts video frames
- Detects human pose landmarks using **MediaPipe**
- Computes key angles (elbows, hips, shoulders)
- Detects the user's **emotional state** from facial features (or voice)
- Uses **NLP** to generate **empathetic and actionable feedback**
- Visualizes all this with **OpenCV**

This project can be extended for:
- Sports performance analysis
- Fitness apps
- Real-time movement coaching
- Rehab and physiotherapy tools

---

## 🚀 Features

### 🕺 Pose Detection with MediaPipe
- Real-time human pose estimation
- Extracts landmarks from frames
- Calculates joint angles (elbow, hip, shoulder)

### 🎥 Video Frame Extraction
- Uses OpenCV to break down any video
- Adjustable frame sampling rate
- Saves extracted frames for analysis

### 🧠 NLP-Driven Advice System
- Analyzes angles and posture
- Provides natural, interpretable feedback using simple NLP logic
- Advice adapts to physical movement style

### 😊 Emotion-Aware Response
- (Optional) Detects user emotion from video or audio
- Feedback tone adjusts: encouraging for frustrated users, detailed for neutral/happy ones

### 🖼️ Annotated Output
- Saves images with drawn landmarks and angle overlays
- Allows visual comparison of posture before and after feedback

---

## 🧰 Technologies Used

| Domain           | Tools Used                            |
|------------------|----------------------------------------|
| Pose Estimation  | [MediaPipe](https://mediapipe.dev)     |
| Computer Vision  | [OpenCV](https://opencv.org)           |
| Math/Numerical   | [NumPy](https://numpy.org)             |
| NLP              | Built-in string logic, OpenAI (ext.)   |
| Emotion Detection| Custom / third-party models (ext.)     |
| Language         | Python 3.x                             |


---

## 🛠️ Installation

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/emotion-pose-feedback.git
cd emotion-pose-feedback
pip install -r requirements.txt
pip install mediapipe opencv-python numpy
pip install fer  # or any facial emotion recognition package

📌 Future Enhancements
🎯 Real-time webcam support

🧠 Deep learning models for advanced advice generation

💬 Voice feedback using text-to-speech

📊 Exporting user progress reports (PDF/CSV)

🕹️ Interactive web interface (Streamlit or Flask)

🤝 Contributing
Contributions are welcome! You can:

Open issues

Submit pull requests

Suggest new use cases

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

👤 Author
Ramy Lazghab
📧 ramy.lazghab@dauphine.tn
