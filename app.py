import streamlit as st
from exactly_one_face import calculate_red_flag_percentage
from eye_gaze_tracker import calculate_attention_percentage
from emotion_tracker import calculate_emotion_percentages
from gesture_posture_tracker import GesturePostureTracker
from pitch_variation_tracker import calculate_pitch_variation_percentages
from llm_feedback import generate_llm_feedback

import tempfile
import os
import json

st.set_page_config(page_title="Student Video Analyzer", layout="centered")

st.title("🎓 Student Video Analyzer")
st.write("Upload a student presentation video. This tool will automatically detect:")
st.markdown("""
- 🎭 Red Flag Detection (face count)
- 👀 Eye gaze attention
- 😊 Emotion distribution
- 🧍‍♂️ Posture and gestures
- 🎙️ Pitch tone variation
- 📝 Transcript with timestamps
- 🤖 LLM Feedback & Score
""")

uploaded_file = st.file_uploader("Upload a video file (e.g., .mp4)", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        temp_video_path = tmp.name
        print(f"[DEBUG] Temp video path: {temp_video_path}")

    st.video(temp_video_path)
    st.info("⏳ Processing video... This might take a while.")

    audio_path = temp_video_path.replace(".mp4", "_audio.wav")

    if not os.path.exists(audio_path):
        from moviepy import VideoFileClip
        print("[DEBUG] Extracting audio...")
        video = VideoFileClip(temp_video_path)
        video.audio.write_audiofile(audio_path)
        video.close()
        print("[DEBUG] Audio extracted successfully.")

    result_data = {}

    try:
        import whisper_timestamped as whisper
        print("[DEBUG] Loading Whisper model...")
        model = whisper.load_model("base")
        transcription_result = whisper.transcribe(model, audio_path)
        print("[DEBUG] Transcription completed.")

        transcript_data = []
        for segment in transcription_result["segments"]:
            transcript_data.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        transcript_path = os.path.join(os.path.dirname(temp_video_path), "transcript_with_timestamps.json")
        print(f"[DEBUG] Saving transcript to: {transcript_path}")
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=4)
        print("[DEBUG] Transcript saved.")

        red_flag_percent = calculate_red_flag_percentage(temp_video_path)
        result_data["red_flag_percentage"] = red_flag_percent
        st.subheader("🚨 Red Flag Detection (Face Count)")
        st.success(f"Red flag percentage: {red_flag_percent:.2f}%")
        if red_flag_percent > 0:
            st.warning("⚠️ Some frames had no face or multiple faces.")
        else:
            st.success("✅ Perfect! Exactly one face was detected in all frames.")

        attention_percent = calculate_attention_percentage(temp_video_path)
        result_data["attention_percentage"] = attention_percent
        st.subheader("🧠 Attention Detection (Eye Gaze)")
        st.success(f"Attention percentage: {attention_percent:.2f}%")
        if attention_percent > 75:
            st.success("✅ Excellent attention!")
        elif attention_percent > 40:
            st.warning("⚠️ Moderate attention. Some distractions detected.")
        else:
            st.error("🚨 Low attention. The student looked away too often.")

        emotion_percentages = calculate_emotion_percentages(temp_video_path)
        result_data["emotion_distribution"] = emotion_percentages
        st.subheader("😊 Emotion Distribution")
        st.bar_chart(emotion_percentages)
        for emotion, percent in emotion_percentages.items():
            st.write(f"**{emotion.capitalize()}**: {percent:.2f}%")

        tracker = GesturePostureTracker()
        gesture_posture_percentages = tracker.calculate_posture_gesture_percentages(temp_video_path)
        result_data["gesture_posture_distribution"] = gesture_posture_percentages
        st.subheader("🧍‍♂️ Body Gesture & Posture")
        st.bar_chart(gesture_posture_percentages)
        for label, percent in gesture_posture_percentages.items():
            st.write(f"**{label}**: {percent:.2f}%")

        pitch_categories = calculate_pitch_variation_percentages(audio_path)
        result_data["pitch_variation_distribution"] = pitch_categories
        st.subheader("🎙️ Pitch Tone Variation (Voice Modulation)")
        if "Error" in pitch_categories:
            st.error(f"Error in pitch analysis: {pitch_categories['Error']}")
        else:
            st.bar_chart(pitch_categories)
            for tone, percent in pitch_categories.items():
                st.write(f"**{tone}**: {percent:.2f}%")

        result_data["transcript"] = transcript_data
        st.subheader("📝 Transcript with Timestamps")
        for segment in transcript_data:
            st.write(f"**[{segment['start']:.2f}s - {segment['end']:.2f}s]**: {segment['text']}")

        result_json_path = os.path.join(os.path.dirname(temp_video_path), "results_summary.json")
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

        st.success("✅ Analysis complete. Summary saved to `results_summary.json`.")

        st.subheader("🤖 LLM Feedback")

        llm_feedback = generate_llm_feedback(
            metrics_path=result_json_path,
            transcript_path=transcript_path
        )

        # llm_feedback is already a dict (parsed JSON), so pass directly to st.json
        if isinstance(llm_feedback, dict):
            st.json(llm_feedback, expanded=True)
            # Debug prints for inspection
            print("\n\n[DEBUG] LLM Feedback (parsed):")
            print(llm_feedback)
        else:
            # If not dict (unlikely), fallback to text area
            st.text_area("LLM Feedback & Scoring", value=str(llm_feedback), height=500)

    except Exception as e:
        print(f"[ERROR] Exception during processing: {e}")
        st.error(f"❌ Error processing video: {e}")

    try:
        print("[DEBUG] Cleaning up temp files...")
        os.remove(temp_video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        print("[DEBUG] Cleanup complete.")
    except Exception as cleanup_error:
        print(f"[WARNING] Cleanup error: {cleanup_error}")
        st.warning(f"⚠️ Cleanup warning: {cleanup_error}")
