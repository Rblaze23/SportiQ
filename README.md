# 🏃‍♂️ Pose Analysis and Feedback System

A computer vision project using **MediaPipe** and **OpenCV** to extract human pose landmarks from video frames and generate insightful feedback based on key joint angles. Ideal for analyzing athletic performance or movement quality.

## 📸 Features

- Extract frames from any video at a custom frame rate
- Detect human pose landmarks using MediaPipe
- Calculate angles at elbows, knees, and shoulders
- Visualize pose skeletons and joints on each frame
- Provide actionable feedback on posture and movement

---

## 📁 Project Structure
.
├── pose_utils.py # Functions for pose detection and angle analysis
├── video_utils.py # Video frame extraction at desired FPS
├── input_video.mp4 # Your source video file (you provide this)
├── extracted_frames/ # Auto-created folder for raw frames
├── annotated_frames/ # Auto-created folder for frames with drawn pose and feedback
└── main.py # (Optional) Integrate both utils for full pipeline

---

## 🧰 Technologies Used

- [Python 3.x](https://www.python.org/)
- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

---

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pose-analysis-feedback.git
   cd pose-analysis-feedback
pip install -r requirements.txt
pip install mediapipe opencv-python numpy
🤝 Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

---
