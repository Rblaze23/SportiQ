import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video at a specified frame rate (frames per second).

    Args:
        video_path (str): Path to input video file.
        output_dir (str): Directory to save extracted frames.
        frame_rate (float): Number of frames to extract per second of video.

    Returns:
        int: Number of frames saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("Video FPS is zero, cannot process frames.")

    interval = int(round(fps / frame_rate))
    if interval == 0:
        interval = 1  # to avoid division by zero or no frame saved

    frame_count = 0
    saved_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_name = f"frame_{saved_frames:03d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            saved_frames += 1

        frame_count += 1

    cap.release()
    return saved_frames
