"""
Hand Detector Module
Handles hand detection and landmark extraction using MediaPipe
"""

import cv2
import mediapipe as mp
from utils.drawing_utils import draw_hand_landmarks, draw_bounding_box


class HandDetector:
    """
    Hand detector class using MediaPipe Hands solution.
    """

    def __init__(self, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize the hand detector.

        Args:
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum detection confidence threshold
            tracking_confidence: Minimum tracking confidence threshold
        """
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )

    def find_hands(self, frame, draw=True):
        """
        Detect hands in the frame and optionally draw landmarks.

        Args:
            frame: BGR image frame from OpenCV
            draw: Whether to draw landmarks on the frame

        Returns:
            frame: Processed frame with drawings (if enabled)
            hands: List of detected hands with landmark data
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hands = []

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand info (left or right)
                hand_type = results.multi_handedness[hand_idx].classification[0].label

                # Extract landmark positions
                landmarks = []
                h, w, _ = frame.shape
                x_coords = []
                y_coords = []

                for lm in hand_landmarks.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    landmarks.append((px, py))
                    x_coords.append(px)
                    y_coords.append(py)

                # Calculate bounding box
                bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

                # Store hand data
                hands.append({
                    "landmarks": landmarks,
                    "bbox": bbox,
                    "type": hand_type
                })

                # Draw on frame if enabled
                if draw:
                    draw_hand_landmarks(frame, landmarks)
                    draw_bounding_box(frame, bbox, hand_type)

        return frame, hands

    def get_finger_tips(self, hand):
        """
        Get the positions of all finger tips.

        Args:
            hand: Hand data dictionary from find_hands()

        Returns:
            Dictionary with finger tip positions
        """
        landmarks = hand["landmarks"]
        return {
            "thumb": landmarks[4],
            "index": landmarks[8],
            "middle": landmarks[12],
            "ring": landmarks[16],
            "pinky": landmarks[20]
        }

    def count_fingers(self, hand):
        """
        Count the number of raised fingers.

        Args:
            hand: Hand data dictionary from find_hands()

        Returns:
            Number of raised fingers (0-5)
        """
        landmarks = hand["landmarks"]
        fingers = []

        # Thumb (compare x position based on hand type)
        if hand["type"] == "Right":
            fingers.append(1 if landmarks[4][0] < landmarks[3][0] else 0)
        else:
            fingers.append(1 if landmarks[4][0] > landmarks[3][0] else 0)

        # Other fingers (compare y position - tip vs pip)
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        for tip, pip in zip(finger_tips, finger_pips):
            fingers.append(1 if landmarks[tip][1] < landmarks[pip][1] else 0)

        return sum(fingers)

    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()
