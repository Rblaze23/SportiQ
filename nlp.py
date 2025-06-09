from transformers import pipeline, set_seed
import logging
import numpy as np

# Configure logging to file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nlp.log'),
        logging.StreamHandler()  # Print to console
    ]
)

generator = None
try:
    logging.info("Loading distilgpt2 model...")
    generator = pipeline("text-generation", model="distilgpt2", device=-1)
    set_seed(42)
    logging.info("distilgpt2 model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load distilgpt2: {str(e)}")
    generator = None

def generate_coach_feedback(data, max_length=150, temperature=0.7):
    """
    Generate tennis coaching feedback based on frame analysis data.

    Args:
        data: List of frame feedback dictionaries or string prompt.
        max_length (int): Maximum length of generated text.
        temperature (float): Sampling temperature for creativity.

    Returns:
        str: Coaching feedback text.
    """
    logging.info("Starting generate_coach_feedback function.")

    # Validate and process input data
    if isinstance(data, list) and data:
        logging.info(f"Received frame_feedback with {len(data)} frames.")
        try:
            elbow_angles = [f.get('elbow_angle') for f in data if f.get('elbow_angle') is not None]
            hip_angles = [f.get('hip_angle') for f in data if f.get('hip_angle') is not None]
            emotions = [f.get('emotion') for f in data if f.get('emotion')]
            
            avg_elbow = np.nanmean(elbow_angles) if elbow_angles else float('nan')
            avg_hip = np.nanmean(hip_angles) if hip_angles else float('nan')
            
            logging.info(f"Computed: avg_elbow={avg_elbow:.2f}, avg_hip={avg_hip:.2f}, emotions={set(emotions)}")

            prompt = (
                "As a professional tennis coach, provide concise, actionable, and motivational feedback for a player's training session based on this data:\n"
                f"- Average Elbow Angle: {avg_elbow:.2f}°\n" if not np.isnan(avg_elbow) else "- Average Elbow Angle: Not available\n"
                f"- Average Hip Angle: {avg_hip:.2f}°\n" if not np.isnan(avg_hip) else "- Average Hip Angle: Not available\n"
                f"- Emotions Detected: {', '.join(set(emotions)) if emotions else 'None'}\n"
                "Summarize performance and give specific improvement tips in a positive tone."
            )

            # Fallback feedback
            fallback = "🎾 Great effort on the court! 💪\n"
            if not np.isnan(avg_elbow):
                if avg_elbow > 120:
                    fallback += "- Bend your elbow more during serves for extra power! 🚀\n"
                elif avg_elbow < 110:
                    fallback += "- Extend your elbow slightly for better groundstroke reach! 🎯\n"
                else:
                    fallback += "- Your elbow angle is perfect! Keep it steady for precision! 🌟\n"
            else:
                fallback += "- Focus on consistent elbow positioning for stronger shots! 🎾\n"
            if not np.isnan(avg_hip):
                if avg_hip > 90:
                    fallback += "- Lower your hips for better balance in rallies! ⚖️\n"
                elif avg_hip < 80:
                    fallback += "- Engage your hips more for explosive power! 💥\n"
                else:
                    fallback += "- Awesome hip form! Keep it up for powerful swings! 🏆\n"
            else:
                fallback += "- Work on hip rotation for more shot power! 💪\n"
            if emotions and ("happy" in emotions or "neutral" in emotions):
                fallback += "- Your positive vibe is a winner! Keep that energy! 😄\n"
            else:
                fallback += "- Stay positive to boost your game! You’ve got this! 🏅\n"
            fallback += "Practice regularly to become a tennis star!"
        except Exception as e:
            logging.error(f"Error processing frame_feedback: {str(e)}")
            fallback = "🎾 Keep swinging! Work on form and positivity to level up! 💪"
    else:
        logging.warning(f"Invalid or empty input data: {type(data)}")
        prompt = str(data) if data else "Provide general tennis coaching tips."
        fallback = "🎾 Awesome work! Focus on steady form and a positive mindset to shine! 💪"

    # Check model availability
    if generator is None:
        logging.warning("Model not loaded, returning fallback feedback.")
        return fallback

    # Attempt generation
    try:
        logging.info(f"Generating feedback with prompt: {prompt[:100]}...")
        results = generator(
            prompt,
            max_length=len(prompt.split()) + max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=generator.tokenizer.eos_token_id,
            truncation=True
        )
        feedback = results[0]['generated_text'][len(prompt):].strip()
        logging.info(f"Generated feedback: {feedback[:100]}...")
        return feedback if feedback else fallback
    except Exception as e:
        logging.error(f"Feedback generation failed: {str(e)}")
        return fallback