from deepface import DeepFace
import cv2
import numpy as np

def detect_emotion(frame):
    try:
        # Convert BGR (OpenCV) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Analyze emotions
        results = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        
        if not results or not isinstance(results, list):
            return None, None
        
        # Get dominant emotion and confidence
        dominant_emotion = results[0]['dominant_emotion']
        confidence = results[0]['emotion'][dominant_emotion] / 100  # Normalize to 0-1
        return dominant_emotion, confidence

    except Exception as e:
        print(f"Emotion detection error: {e}")
        return None, None