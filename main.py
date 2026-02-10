import cv2
import time
from hand_detector import HandDetector
from utils.drawing_utils import draw_hand_points, draw_bounding_box, draw_hand_skeleton

# Initialize hand detector
detector = HandDetector(max_hands=2, detection_confidence=0.7, tracking_confidence=0.5)

# Initialize webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Set frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if camera opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Initialize variables
prev_time = 0

# Open webcam & reading frames
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detect hands using HandDetector
    hands_data = detector.detect(frame)
    
    # Process each detected hand
    for hand in hands_data:
        hand_type = hand["label"]
        positions = hand["positions"]
        hand_landmarks = hand["landmarks"]
        
        # Draw hand landmarks (21 points)
        draw_hand_points(frame, positions)
        
        # Draw connections between landmarks
        draw_hand_skeleton(frame, hand_landmarks, detector.mp_hands, detector.mp_drawing)
        
        # Draw bounding box with label
        box_color = (255, 0, 0) if hand_type == "Left" else (0, 0, 255)
        label = f"{hand_type} Hand"
        draw_bounding_box(frame, positions, label, box_color)

    # Calculate fps
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display fps
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop if ESC is pressed or window X button is clicked
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty('Hand Tracking', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()