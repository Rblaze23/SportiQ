import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose


def get_pose_landmarks(frame):
    """
    Detect pose landmarks in a static image frame.
    Args:
        frame: Image in BGR format (from OpenCV)
    Returns:
        List of dicts with 'x', 'y', 'z', 'visibility' or None if no landmarks
    """
    with mp_pose.Pose(static_image_mode=True) as pose:
        # Convert BGR image to RGB for MediaPipe compatibility
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image to get pose landmarks
        results = pose.process(image_rgb)

        # If no pose detected, return None
        if not results.pose_landmarks:
            return None

        # Extract each landmark's x, y, z, visibility into a dictionary
        landmarks = [{
            'x': lm.x,
            'y': lm.y,
            'z': lm.z,
            'visibility': lm.visibility
        } for lm in results.pose_landmarks.landmark]

        return landmarks


def calculate_angle(a, b, c):
    """
    Calculates the angle at point b formed by points a and c.
    Returns angle in degrees.
    """
    # Convert dicts to numpy 2D coordinates
    a = np.array([a['x'], a['y']])
    b = np.array([b['x'], b['y']])
    c = np.array([c['x'], c['y']])

    # Vectors from b to a and b to c
    ba = a - b
    bc = c - b

    # Use cosine formula to get angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def detect_pose(frame, landmarks):
    """
    Draws pose skeleton and keypoints on a frame.
    """
    if landmarks is None:
        return frame

    height, width, _ = frame.shape

    # Define key body part connections to draw the skeleton
    connections = [
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 12),            # Shoulders
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
        (23, 24)             # Hips
    ]

    # Convert normalized coords to pixel positions and ignore low-visibility points
    landmark_points = {
        idx: (int(lm['x'] * width), int(lm['y'] * height))
        for idx, lm in enumerate(landmarks)
        if lm['visibility'] > 0.5
    }

    # Draw lines between keypoints
    for start_idx, end_idx in connections:
        if start_idx in landmark_points and end_idx in landmark_points:
            cv2.line(frame, landmark_points[start_idx],
                     landmark_points[end_idx], (0, 255, 0), 2)

    # Draw circles for detected keypoints
    for point in landmark_points.values():
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

    return frame


def generate_advice(landmarks):
    if landmarks is None or len(landmarks) < 29:
        return None, None, "No valid pose detected."

    # Left side example
    elbow_angle = calculate_angle(
        landmarks[11], landmarks[13], landmarks[15])  # Shoulder, Elbow, Wrist
    hip_angle = calculate_angle(
        landmarks[11], landmarks[23], landmarks[25])    # Shoulder, Hip, Knee

    advice = ""

    if elbow_angle is not None:
        if elbow_angle < 70:
            advice += "Try extending your arm more during the swing. "
        elif elbow_angle > 160:
            advice += "Elbow overextended. Consider maintaining a controlled follow-through. "
        else:
            advice += "Elbow angle looks good. "

    if hip_angle is not None:
        if hip_angle < 100:
            advice += "Bend your knees and rotate hips more for power. "
        elif hip_angle > 160:
            advice += "Over-rotation of hips. Stabilize your core. "
        else:
            advice += "Good hip rotation. "

    return elbow_angle, hip_angle, advice


def midpoint(p1, p2):
    return {
        'x': (p1['x'] + p2['x']) / 2,
        'y': (p1['y'] + p2['y']) / 2,
        'z': (p1.get('z', 0) + p2.get('z', 0)) / 2,
        'visibility': min(p1.get('visibility', 1.0), p2.get('visibility', 1.0))
    }


def get_pose_angles(landmarks):
    """
    Calculate key joint angles (elbow, knee, shoulder).

    Args:
        landmarks: List of dicts from get_pose_landmarks

    Returns:
        Dict of named angles
    """
    if landmarks is None or len(landmarks) < 29:
        return {}

    mid_shoulder = midpoint(landmarks[11], landmarks[12])
    print("Args to calculate_angle:", landmarks[11], landmarks[12], mid_shoulder)

    angles = {
        'left_elbow': calculate_angle(landmarks[11], landmarks[13], landmarks[15]),
        'right_elbow': calculate_angle(landmarks[12], landmarks[14], landmarks[16]),
        'left_knee': calculate_angle(landmarks[23], landmarks[25], landmarks[27]),
        'right_knee': calculate_angle(landmarks[24], landmarks[26], landmarks[28]),
        'shoulder_angle': calculate_angle(landmarks[11], landmarks[12], mid_shoulder),
    }
    return angles
