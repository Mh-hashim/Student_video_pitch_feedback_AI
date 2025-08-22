import librosa
import numpy as np
import torch
import crepe

def classify_pitch_range(pitch_range_hz):
    if pitch_range_hz < 20:
        return "Flat/monotone (pitch range < 20 Hz)"
    elif pitch_range_hz <= 60:
        return "Some variation (20–60 Hz)"
    else:
        return "Strong, expressive tone (pitch range > 60 Hz)"

def calculate_pitch_variation_percentages(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)  # CREPE expects 16kHz

    # Run CREPE (use 100ms step size = 10Hz frame rate)
    _, frequency, confidence, _ = crepe.predict(y, sr, viterbi=True, step_size=100)

    # Filter out low confidence predictions
    threshold = 0.5
    valid_freq = frequency[confidence > threshold]

    if len(valid_freq) < 2:
        return {
            "Flat/monotone (pitch range < 20 Hz)": 0.0,
            "Some variation (20–60 Hz)": 0.0,
            "Strong, expressive tone (pitch range > 60 Hz)": 0.0,
        }

    # Calculate pitch range (max - min) over sliding windows
    window_size = 30  # frames (3 seconds)
    step = 10         # frames (1 second)

    counts = {"Flat/monotone (pitch range < 20 Hz)": 0,
              "Some variation (20–60 Hz)": 0,
              "Strong, expressive tone (pitch range > 60 Hz)": 0}

    total_windows = 0
    for i in range(0, len(valid_freq) - window_size + 1, step):
        window = valid_freq[i:i + window_size]
        pitch_range = np.max(window) - np.min(window)
        label = classify_pitch_range(pitch_range)
        counts[label] += 1
        total_windows += 1

    if total_windows == 0:
        return {label: 0.0 for label in counts}

    # Convert to percentages
    percentages = {label: (count / total_windows) * 100 for label, count in counts.items()}
    return percentages
