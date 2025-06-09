import shutil
import cv2
import streamlit as st
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px
import os
import sys
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tennis.log'),
        logging.StreamHandler()
    ]
)

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Custom utility imports
from utils.video_utils import extract_frames
from utils.pose_utils import get_pose_landmarks, detect_pose, generate_advice
from utils.emotion_utils import detect_emotion
from utils.nlp import generate_coach_feedback

# Page Configuration
st.set_page_config(
    page_title="SportIQ - Tennis Analyzer",
    page_icon="🎾",
    layout="wide"
)

# Sidebar Navigation
with st.sidebar:
    try:
        st.image("log1.jpg", width=200)
    except FileNotFoundError:
        st.warning("Logo file not found. Place log1.jpg in the project directory.")
    st.title("SportIQ Dashboard 🎾")
    st.info("AI-Powered Tennis Performance Tracker")
    st.session_state.user_name = st.text_input("Your Name", "Player")
    page = st.selectbox("Select Page", ["Analytics", "Performance Metrics"])

# Page Title
st.title("🎾 SportIQ - Tennis Performance Analyzer")

# Initialize session_state keys
for key in ['frames', 'frame_index', 'elbow_angles', 'uploaded_video_path', 'avg_elbow_angle', 'frame_feedback', 'coach_feedback', 'analysis_done']:
    if key not in st.session_state:
        if key in ['frames', 'elbow_angles', 'frame_feedback']:
            st.session_state[key] = []
        else:
            st.session_state[key] = None if key == 'uploaded_video_path' else False if key == 'analysis_done' else None

# ==Analytics Page==
if page == "Analytics":
    st.header("📈 Upload & Analyze Your Training Video")
    st.markdown("""
    Upload your tennis training video to:
    - Watch it directly here.
    - Automatically extract frames from
    - Detect poses using AI.
    - Analyze elbow and hip angles.
    - Detect emotions.
    - Get improvement tips.
    """)

    uploaded_file = st.file_uploader("Upload a short training video (mp4 or avi4)", type=["mp4", "avi"])
    if uploaded_file:
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_path.write(uploaded_file.read())
        temp_video_path.close()

        if st.session_state.get('uploaded_video_path', '') != temp_video_path.name:
            st.session_state['uploaded_video_path'] = temp_video_path.name
            st.session_state['elbow_angles'] = []
            st.session_state['frame_feedback'] = []
            st.session_state['frames'] = []
            st.session_state['avg_elbow_angle'] = None
            st.session_state['analysis_done'] = False
            st.session_state['coach_feedback'] = None

        st.success("✅ Video uploaded successfully.")

    if st.session_state.get('uploaded_video_path'):
        st.video(st.session_state['uploaded_video_path'])

        if not st.session_state.frames:
            if st.button("Run Pose Analysis"):
                output_dir = tempfile.mkdtemp()
                try:
                    with st.spinner('🔍 Extracting frames and analyzing poses and emotions...'):
                        num_frames = extract_frames(st.session_state['uploaded_video_path'], output_dir)
                        elbow_angles = []
                        frame_feedback = []
                        frames = []

                        for idx, frame_file in enumerate(sorted(os.listdir(output_dir))):
                            frame_path = os.path.join(output_dir, frame_file)
                            frame = cv2.imread(frame_path)
                            try:
                                landmarks = get_pose_landmarks(frame)
                                pose_image = detect_pose(frame.copy(), landmarks)
                                elbow_angle, hip_angle, advice = generate_advice(landmarks)
                                emotion, confidence = detect_emotion(frame)
                                elbow_angles.append(elbow_angle if elbow_angle is not None else np.nan)
                                frame_feedback.append({
                                    "frame": idx,
                                    "elbow_angle": round(elbow_angle, 2) if elbow_angle is not None else None,
                                    "hip_angle": round(hip_angle, 2) if hip_angle is not None else None,
                                    "emotion": emotion,
                                    "emotion_confidence": confidence,
                                    "advice": advice
                                })
                                pose_image_rgb = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
                                frames.append(pose_image_rgb)
                            except Exception as e:
                                st.warning(f"⚠️ Error analyzing frame {idx}: {e}")
                                logging.error(f"Frame {idx} analysis error: {str(e)}")

                        st.session_state.frames = frames
                        st.session_state.frame_feedback = frame_feedback
                        st.session_state.elbow_angles = elbow_angles

                        if elbow_angles:
                            st.session_state.avg_elbow_angle = np.nanmean(elbow_angles)

                        # Log frame_feedback for debugging
                        logging.info(f"Frame feedback structure: {st.session_state.frame_feedback[:2]}")

                        # Generate and clean coach feedback
                        if st.session_state.frame_feedback:
                            try:
                                raw_feedback = generate_coach_feedback(st.session_state.frame_feedback, max_length=150, temperature=0.7)
                                logging.info(f"Raw coach feedback: {raw_feedback[:200]}...")
                                # Clean feedback: remove repetitive "Average Elbow Angle" lines
                                lines = raw_feedback.split('\n')
                                seen = set()
                                cleaned_feedback = []
                                elbow_angle_count = 0
                                for line in lines:
                                    if line.startswith('- Average Elbow Angle'):
                                        elbow_angle_count += 1
                                        if elbow_angle_count == 1:
                                            seen.add(line)
                                            cleaned_feedback.append(line)
                                    elif line not in seen:
                                        seen.add(line)
                                        cleaned_feedback.append(line)
                                st.session_state.coach_feedback = '\n'.join(cleaned_feedback).strip()
                                # If feedback is empty or only contains angles, use fallback
                                if not st.session_state.coach_feedback or all(line.startswith('- Average') for line in cleaned_feedback):
                                    st.session_state.coach_feedback = f"🎾 Keep practicing, {st.session_state.user_name}! Focus on form and positivity. 💪"
                                logging.info(f"Cleaned coach feedback: {st.session_state.coach_feedback[:100]}...")
                                #st.success("✅ Coach feedback generated.")
                            except Exception as e:
                                logging.error(f"Coach feedback generation failed: {str(e)}")
                                st.session_state.coach_feedback = f"🎾 Keep practicing, {st.session_state.user_name}! Focus on form and positivity. 💪"
                        else:
                            st.session_state.coach_feedback = "No feedback available: Analyze more frames."
                        st.session_state.analysis_done = True

                finally:
                    shutil.rmtree(output_dir)

        # Display frames and results
        if st.session_state.get('analysis_done', False) and st.session_state.frames:
            st.subheader("🖼️ All Frames and Feedback")
            st.success(f"✅ {len(st.session_state.frames)} frames extracted.")
            if st.session_state.avg_elbow_angle:
                st.success(f"✅ Average Elbow Angle: {st.session_state.avg_elbow_angle:.2f}°")
            if st.session_state.coach_feedback:
                st.success("✅ Coach feedback generated.")

            for idx, (frame_img, feedback) in enumerate(zip(st.session_state.frames, st.session_state.frame_feedback)):
                st.image(frame_img, caption=f"Pose - Frame {idx + 1}", width=600)
                elbow_angle = f"{feedback['elbow_angle']}°" if feedback['elbow_angle'] is not None else "N/A"
                hip_angle = f"{feedback['hip_angle']}°" if feedback['hip_angle'] is not None else "N/A"
                emotion = feedback['emotion'] if feedback.get('emotion') else "No emotion detected"
                confidence = f"{feedback['emotion_confidence']*100:.1f}%" if feedback.get('emotion_confidence') else "N/A"
                advice = feedback['advice'] if feedback.get('advice') else "No advice available"
                st.markdown(f"""
                - **Elbow Angle**: {elbow_angle}
                - **Hip Angle**: {hip_angle}
                - **Emotion**: {emotion}
                - **Confidence**: {confidence}
                - **Advice**: {advice}
                """)
                st.markdown("---")

            if st.session_state.coach_feedback:
                st.subheader("🏅 Coach's Summary Feedback")
                st.markdown(st.session_state.coach_feedback)

            df = pd.DataFrame(st.session_state.frame_feedback)
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Feedback Report",
                data=csv_data,
                file_name="pose_feedback.csv",
                mime="text/csv"
            )

# Performance Metrics Page
elif page == "Performance Metrics":
    st.header("📊 Key Performance Metrics 🎾")
    st.markdown(f"Dive into your tennis performance, {st.session_state.user_name}, with detailed metrics and pro-level insights to elevate your game! 🚀")

    # Initialize variables
    angles_np = None
    avg_elbow = None
    std_elbow = None
    max_elbow = None
    min_elbow = None
    hip_angles_np = None
    avg_hip = None
    std_hip = None
    max_hip = None
    min_hip = None
    emotions = []
    emotion_counts = None
    avg_confidence = None

    # Elbow Angle Metrics
    if st.session_state.get('elbow_angles'):
        angles = st.session_state['elbow_angles']
        angles_np = np.array(angles, dtype=np.float64)
        avg_elbow = np.nanmean(angles_np)
        max_elbow = np.nanmax(angles_np)
        min_elbow = np.nanmin(angles_np)
        std_elbow = np.nanstd(angles_np)

    # Hip Angle Metrics
    if st.session_state.get('frame_feedback'):
        hip_angles = [f.get('hip_angle') for f in st.session_state['frame_feedback'] if f.get('hip_angle') is not None]
        hip_angles_np = np.array(hip_angles, dtype=np.float64) if hip_angles else np.array([])
        avg_hip = np.nanmean(hip_angles_np) if hip_angles_np.size else np.nan
        max_hip = np.nanmax(hip_angles_np) if hip_angles_np.size else np.nan
        min_hip = np.nanmin(hip_angles_np) if hip_angles_np.size else np.nan
        std_hip = np.nanstd(hip_angles_np) if hip_angles_np.size else np.nan
        emotions = [f.get('emotion') for f in st.session_state['frame_feedback'] if f.get('emotion')]
        confidences = [f.get('emotion_confidence') for f in st.session_state['frame_feedback'] if f.get('emotion_confidence') is not None]
        emotion_counts = pd.Series(emotions).value_counts() if emotions else None
        avg_confidence = np.mean([c * 100 for c in confidences]) if confidences else None

    # Display Elbow Metrics
    if angles_np is not None and len(angles_np) > 0 and not np.isnan(avg_elbow):
        st.subheader("🎾 Elbow Angle Analysis")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Elbow Angle", f"{avg_elbow:.2f}°")
        col2.metric("Max Elbow Angle", f"{max_elbow:.2f}°")
        col3.metric("Min Elbow Angle", f"{min_elbow:.2f}°")
        col4.metric("Elbow Variability", f"{std_elbow:.2f}°")

        df_elbow = pd.DataFrame({
            "Frame #": list(range(1, len(angles_np) + 1)),
            "Elbow Angle (°)": angles_np
        })
        fig_elbow = px.line(df_elbow, x="Frame #", y="Elbow Angle (°)", markers=True, title="Elbow Angle Over Time")
        fig_elbow.update_traces(line_color="#4CAF50", marker=dict(size=8))
        st.plotly_chart(fig_elbow, use_container_width=True)

        st.markdown("📖 **Elbow Pro Tip** 🎯")
        if std_elbow < 10:
            st.success(f"🎾 {st.session_state.user_name}, your elbow is rock-solid! 🔥 Keep that steady angle for pinpoint serves and volleys. Pro move! 💪")
        elif std_elbow < 20:
            st.info(f"🎵 {st.session_state.user_name}, nice elbow work, but it’s dancing a bit! 🎶 Try a smoother arm motion for better shot control. You’ve got this! 😎")
        else:
            st.warning(f"🎾 {st.session_state.user_name}, whoa, your elbow’s got some flair! 🌟 Lock in a consistent angle during serves and forehands to ace your shots. 🏆")

    else:
        st.info("⚠️ No elbow data yet. Head to the Analytics page to analyze your video and unlock your metrics! 📹")

    # Display Hip Metrics
    if hip_angles_np is not None and len(hip_angles_np) > 0 and not np.isnan(avg_hip):
        st.subheader("🎾 Hip Angle Analysis")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Hip Angle", f"{avg_hip:.2f}°")
        col2.metric("Max Hip Angle", f"{max_hip:.2f}°")
        col3.metric("Min Hip Angle", f"{min_hip:.2f}°")
        col4.metric("Hip Variability", f"{std_hip:.2f}°")

        df_hip = pd.DataFrame({
            "Frame #": list(range(1, len(hip_angles_np) + 1)),
            "Hip Angle (°)": hip_angles_np
        })
        fig_hip = px.line(df_hip, x="Frame #", y="Hip Angle (°)", markers=True, title="Hip Angle Over Time")
        fig_hip.update_traces(line_color="#2196F3", marker=dict(size=8))
        st.plotly_chart(fig_hip, use_container_width=True)

        st.markdown("📖 **Hip Pro Tip** 🎯")
        if std_hip < 10:
            st.success(f"🎾 {st.session_state.user_name}, your hips are a powerhouse! 💥 Stable and strong, perfect for crushing groundstrokes. Keep it up! 🌟")
        elif std_hip < 20:
            st.info(f"🎾 {st.session_state.user_name}, solid hip game, but a tad wobbly! ⚖️ Smooth out your rotation for better balance and power. Go for it! 🚀")
        else:
            st.warning(f"🎾 {st.session_state.user_name}, hips are moving like a salsa dance! 💃 Anchor them for steadier shots and unstoppable rallies. You can do it! 🏅")

    else:
        st.info("⚠️ No hip data yet. Analyze a video on the Analytics page to see your metrics! 📹")

    # Display Emotion Metrics
    if emotion_counts is not None and len(emotions) > 0:
        st.subheader("🎾 Emotion Analysis")
        df_emotions = pd.DataFrame({
            "Emotion": emotion_counts.index,
            "Count": emotion_counts.values
        })
        fig_emotion = px.bar(df_emotions, x="Emotion", y="Count", title="Emotion Distribution Across Frames", color="Emotion")
        st.plotly_chart(fig_emotion, use_container_width=True)

        if avg_confidence:
            st.metric("Average Emotion Confidence", f"{avg_confidence:.1f}%")

        st.markdown("📖 **Mindset Pro Tip** 🎯")
        if "happy" in emotions or "neutral" in emotions:
            st.success(f"🎾 {st.session_state.user_name}, you’re radiating focus and positivity! 😄 Stay in this zone to dominate the court with confidence. Smash it! 🚀")
        elif "angry" in emotions or "sad" in emotions:
            st.warning(f"🎾 {st.session_state.user_name}, tough vibes detected. 😣 Take a deep breath and channel that energy into your next killer shot. You’ve got the heart of a champion! 💪")
        else:
            st.info(f"🎾 {st.session_state.user_name}, mixed emotions on the court. 🧘 Stay mindful and keep your cool to unlock your full potential. Let’s go! 🏆")

    else:
        st.info("⚠️ No emotion data yet. Analyze a video to uncover your mindset insights! 📹")

    # Performance Summary
    st.markdown("---")
    st.subheader("🏆 Your Tennis Performance Snapshot")
    if (angles_np is not None and not np.isnan(avg_elbow)) and (hip_angles_np is not None and not np.isnan(avg_hip)):
        elbow_score = 10 * max(0, 1 - min(std_elbow, 40) / 40)
        hip_score = 10 * max(0, 1 - min(std_hip, 40) / 40)
        elbow_angle_bonus = 3 * max(0, 1 - abs(avg_elbow - 115) / 15) if 100 <= avg_elbow <= 130 else 0
        emotion_bonus = 3 if emotions and ("happy" in emotions or "neutral" in emotions) else 0
        performance_score = elbow_score + hip_score + elbow_angle_bonus + emotion_bonus
        performance_score = min(max(performance_score, 12), 20)  # Minimum score of 12

        col1, col2 = st.columns(2)
        col1.metric("Performance Score", f"{performance_score:.1f}/20")
        col2.metric("Target Elbow Angle", "110-120°", delta=f"{avg_elbow-115:.2f}° from 115°")

        if performance_score > 16:
            st.success(f"🎾 {st.session_state.user_name}, you’re playing like a pro! 🌟 Your form is tight, and you’re ready to ace the competition. Keep shining! ✨")
            st.balloons()
        elif performance_score > 12:
            st.info(f"🎾 {st.session_state.user_name}, strong game! 🏅 Polish your elbow and hip consistency to climb to the next level. You’re almost there! 😎")
        else:
            st.info(f"🎾 {st.session_state.user_name}, you’re on the rise! 🌟 Focus on steady angles and a positive vibe to boost your score. You’ve got this! 🚀")

        df_metrics = pd.DataFrame({
            "Frame #": list(range(1, len(angles_np) + 1)),
            "Elbow Angle (°)": angles_np,
            "Hip Angle (°)": hip_angles_np if len(hip_angles_np) == len(angles_np) else [None] * len(angles_np),
            "Emotion": emotions[:len(angles_np)] + [None] * max(0, len(angles_np) - len(emotions))
        })
        csv_report = df_metrics.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Performance Report",
            data=csv_report,
            file_name="tennis_performance_report.csv",
            mime="text/csv"
        )

    else:
        st.info("⚠️ Need more data for a full snapshot. Run an analysis on the Analytics page! 📊")

    # Progress Tracker
    st.markdown("---")
    st.subheader("🌟 Your Progress Journey")
    if (angles_np is not None and not np.isnan(avg_elbow)) and (hip_angles_np is not None and not np.isnan(avg_hip)):
        elbow_score = 10 * max(0, 1 - min(std_elbow, 40) / 40)
        hip_score = 10 * max(0, 1 - min(std_hip, 40) / 40)
        elbow_angle_bonus = 3 * max(0, 1 - abs(avg_elbow - 115) / 15) if 100 <= avg_elbow <= 130 else 0
        emotion_bonus = 3 if emotions and ("happy" in emotions or "neutral" in emotions) else 0
        performance_score = elbow_score + hip_score + elbow_angle_bonus + emotion_bonus
        performance_score = min(max(performance_score, 12), 20)  # Minimum score of 12

        progress = performance_score / 20
        st.progress(progress)
        st.markdown(f"**Progress: {performance_score:.1f}/20** — {st.session_state.user_name}, you’re {progress*100:.0f}% toward mastering your tennis form! Keep swinging! 🎾")
        if performance_score > 16:
            st.success(f"🎾 {st.session_state.user_name}, you’re a court conqueror! Your form and mindset are top-notch. Aim for the championship! 🏆")
            st.balloons()
        elif performance_score > 12:
            st.info(f"🎾 {st.session_state.user_name}, solid performance! You’re building a strong game. Tweak your form to go pro! 💪")
        else:
            st.info(f"🎾 {st.session_state.user_name}, you’re climbing the ranks! Focus on steady angles and positivity to soar higher! 🚀")
    else:
        st.info(f"⚠️ Start your journey, {st.session_state.user_name}, by analyzing a video to unlock your progress tracker! 📹")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Ramy Lazghab")

# Notes
st.markdown("These thresholds are heuristic starting points for feedback, based on typical angle variability observed in testing. With more data and expert input, they can be refined for more personalized coaching.")