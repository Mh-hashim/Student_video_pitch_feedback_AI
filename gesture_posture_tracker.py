import cv2
import mediapipe as mp

class GesturePostureTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
    
    def calculate_posture_gesture_percentages(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = 0
        stiff_count = 0
        some_gesture_count = 0
        natural_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            total_frames += 1

            # Convert image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            if results.pose_landmarks:
                # Example logic (adjust with real rules):
                # Use hand/shoulder movement and angles to detect posture
                # This is just a placeholder logic — use better heuristics in real case
                left_hand = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
                right_hand = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]

                movement = abs(left_hand.x - right_hand.x)

                if movement < 0.05:
                    stiff_count += 1
                elif movement < 0.15:
                    some_gesture_count += 1
                else:
                    natural_count += 1

        cap.release()

        if total_frames == 0:
            return {
                "Stiff or no gestures": 0,
                "Some gestures": 0,
                "Natural gestures": 0
            }

        return {
            "Stiff or no gestures": (stiff_count / total_frames) * 100,
            "Some gestures": (some_gesture_count / total_frames) * 100,
            "Natural gestures": (natural_count / total_frames) * 100
        }
