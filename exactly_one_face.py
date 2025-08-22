import cv2
import mediapipe as mp

def calculate_red_flag_percentage(video_path: str) -> float:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)

    total_frames = 0
    red_flag_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get face mesh result
        results = face_mesh.process(rgb_frame)

        # Count how many faces are detected
        face_count = 0
        if results.multi_face_landmarks:
            face_count = len(results.multi_face_landmarks)

        # Red flag condition: not exactly 1 face
        if face_count != 1:
            red_flag_frames += 1

    cap.release()
    face_mesh.close()

    if total_frames == 0:
        return 0.0

    red_flag_percentage = (red_flag_frames / total_frames) * 100
    return red_flag_percentage
