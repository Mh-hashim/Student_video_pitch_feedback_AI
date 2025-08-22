import cv2
from deepface import DeepFace
from collections import Counter

def calculate_emotion_percentages(video_path: str, frame_skip: int = 10) -> dict:
    cap = cv2.VideoCapture(video_path)

    emotion_counts = Counter()
    total_processed = 0

    # Initialize all 7 emotions to 0
    all_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    for emotion in all_emotions:
        emotion_counts[emotion] = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % frame_skip != 0:
            continue

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']
            if dominant_emotion in emotion_counts:
                emotion_counts[dominant_emotion] += 1
            total_processed += 1
        except Exception:
            continue

    cap.release()

    if total_processed == 0:
        return {emotion: 0.0 for emotion in all_emotions}

    # Convert to percentage
    emotion_percentages = {
        emotion: round((count / total_processed) * 100, 2)
        for emotion, count in emotion_counts.items()
    }

    return emotion_percentages
