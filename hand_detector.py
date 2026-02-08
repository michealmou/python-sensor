import cv2
import mediapipe as mp
class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawer = mp.drawing_utils
        def detect(self, frame):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            hands_data = []
            if results.multi_hand_landmarks:
                for hand_landmarks, hand_info in zip(
                  results.multi_hand_landmarks,
                  results.multi_handedness  
                ):
                    hand_label = hand_info.classification[0].label
                    hands_data.append({
                        "label":hand_label,
                        "landmarks": hand_landmarks
                    })
            return hands_data
