# hand detector file
import cv2
import mediapipe as mp
from config import max_hands, detection_confidence, tracking_confidence

class HandDetector:
    def __init__(self, max_hands, detection_confidence, tracking_confidence):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
    
    def detect(self, frame):
        """
        Detect hands in frame and return list of hand data with pixel coordinates.
        Each hand dict contains: label, landmarks (raw), positions (list of (id, x, y))
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        hands_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_label = hand_info.classification[0].label
                
                # Convert landmarks to pixel positions
                positions = []
                for idx, lm in enumerate(hand_landmarks.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    positions.append((idx, px, py))
                
                hands_data.append({
                    "label": hand_label,
                    "landmarks": hand_landmarks,
                    "positions": positions
                })
        
        return hands_data