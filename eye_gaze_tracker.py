import cv2
import mediapipe as mp

def calculate_attention_percentage(video_path: str) -> float:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(video_path)

    total_frames = 0
    attentive_frames = 0
    consecutive_lost = 0
    max_tolerable_loss = 5

    LEFT_IRIS = [474]
    RIGHT_IRIS = [469]
    LEFT_EYE_TOP = 159
    LEFT_EYE_CENTER = 468
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_CENTER = 473
    RIGHT_EYE_BOTTOM = 374

    def vertical_distance(p1, p2):
        return abs(p1[1] - p2[1])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        attentive = True  # assume attentive until proven otherwise

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            def get_px(id): return int(landmarks[id].x * w), int(landmarks[id].y * h)

            # Get left and right eye top-center-bottom points
            lt, lc, lb = get_px(LEFT_EYE_TOP), get_px(LEFT_EYE_CENTER), get_px(LEFT_EYE_BOTTOM)
            rt, rc, rb = get_px(RIGHT_EYE_TOP), get_px(RIGHT_EYE_CENTER), get_px(RIGHT_EYE_BOTTOM)

            # Eye closed logic
            def is_eye_closed(top, center, bottom):
                d1 = vertical_distance(top, center)
                d2 = vertical_distance(center, bottom)
                d3 = vertical_distance(top, bottom)
                threshold = 3  # adjust based on resolution
                return d1 < threshold and d2 < threshold and d3 < threshold

            if is_eye_closed(lt, lc, lb) and is_eye_closed(rt, rc, rb):
                consecutive_lost += 1
                if consecutive_lost >= max_tolerable_loss:
                    attentive = False
            else:
                consecutive_lost = 0  # regained attention

        else:
            attentive = False

        if attentive:
            attentive_frames += 1

    cap.release()
    face_mesh.close()

    if total_frames == 0:
        return 0.0

    return (attentive_frames / total_frames) * 100
