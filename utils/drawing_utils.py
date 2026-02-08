"""
Drawing Utilities
Helper functions for drawing hand landmarks and visualizations
"""

import cv2


# Hand landmark connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17)
]

# Colors (BGR format)
LANDMARK_COLOR = (0, 255, 255)      # Yellow
CONNECTION_COLOR = (0, 255, 0)       # Green
BBOX_COLOR_LEFT = (255, 0, 0)        # Blue
BBOX_COLOR_RIGHT = (0, 0, 255)       # Red
TEXT_COLOR = (255, 255, 255)         # White


def draw_hand_landmarks(frame, landmarks, draw_connections=True):
    """
    Draw hand landmarks and connections on the frame.

    Args:
        frame: OpenCV image (BGR)
        landmarks: List of (x, y) landmark positions
        draw_connections: Whether to draw lines between landmarks
    """
    # Draw connections first (so landmarks appear on top)
    if draw_connections:
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                cv2.line(frame, start_point, end_point, CONNECTION_COLOR, 2)

    # Draw landmarks
    for idx, (x, y) in enumerate(landmarks):
        # Fingertips get larger circles
        radius = 8 if idx in [4, 8, 12, 16, 20] else 5
        cv2.circle(frame, (x, y), radius, LANDMARK_COLOR, cv2.FILLED)
        cv2.circle(frame, (x, y), radius, (0, 0, 0), 1)  # Black outline


def draw_bounding_box(frame, bbox, hand_type="Unknown", padding=20):
    """
    Draw a bounding box around the hand with label.

    Args:
        frame: OpenCV image (BGR)
        bbox: Tuple of (x_min, y_min, x_max, y_max)
        hand_type: "Left" or "Right"
        padding: Extra padding around the box
    """
    x_min, y_min, x_max, y_max = bbox

    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = x_max + padding
    y_max = y_max + padding

    # Choose color based on hand type
    color = BBOX_COLOR_LEFT if hand_type == "Left" else BBOX_COLOR_RIGHT

    # Draw rectangle
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    # Draw label background
    label = f"{hand_type} Hand"
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x_min, y_min - 25), (x_min + text_width + 10, y_min), color, cv2.FILLED)

    # Draw label text
    cv2.putText(frame, label, (x_min + 5, y_min - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)


def draw_finger_count(frame, count, position=(50, 100)):
    """
    Display the finger count on the frame.

    Args:
        frame: OpenCV image (BGR)
        count: Number of fingers (0-5)
        position: (x, y) position for the text
    """
    cv2.putText(frame, f"Fingers: {count}", position,
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)


def draw_fps(frame, fps, position=(10, 30)):
    """
    Display FPS on the frame.

    Args:
        frame: OpenCV image (BGR)
        fps: Frames per second value
        position: (x, y) position for the text
    """
    cv2.putText(frame, f"FPS: {int(fps)}", position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
