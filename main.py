import cv2
import time
import math
import numpy as np
from hand_detector import HandDetector
from utils.drawing_utils import draw_hand_points, draw_bounding_box, draw_hand_skeleton
from mouse_controller import MouseController

# Initialize MouseController
if MouseController:
    mouse = MouseController(smoothing_factor=6)
else:
    mouse = None

# Frame Reduction (padding from the edge of the webcam view)
frameR = 150

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
    
    # Process ONLY the first detected hand to avoid cursor conflict
    if hands_data:
        hand = hands_data[0] # Focus on the first hand
        positions = hand["positions"]
        hand_landmarks = hand["landmarks"]
        
        # Draw Visuals
        draw_hand_points(frame, positions)
        draw_hand_skeleton(frame, hand_landmarks, detector.mp_hands, detector.mp_drawing)
        
        if positions:
            # EXTRACT LANDMARKS
            # Index Finger Tip (8)
            x1, y1 = positions[8][1], positions[8][2]
            # Thumb Tip (4)
            x2, y2 = positions[4][1], positions[4][2]
            
            # Draw circles for key fingers
            cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            
            # 1. MOVE MOUSE (Map coordinates)
            if mouse:
                # Map x1 from [frameR, 1280-frameR] to [0, screenWidth]
                x3 = np.interp(x1, (frameR, 1280 - frameR), (0, mouse.screen_width))
                y3 = np.interp(y1, (frameR, 720 - frameR), (0, mouse.screen_height))
                mouse.move(x3, y3)

            # 2. CLICK GESTURE
            distance = math.hypot(x2 - x1, y2 - y1)
            if distance < 40:
                cv2.circle(frame, (x1, y1), 15, (0, 255, 0), cv2.FILLED) # Green feedback
                if mouse:
                    mouse.click('left')
                    time.sleep(0.1) # Debounce equivalent
            
            # 3. RIGHT CLICK GESTURE
            # Middle Finger Tip (12)
            x3, y3 = positions[12][1], positions[12][2]
            
            # Calculate distance between Thumb (4) and Middle (12)
            distance_right = math.hypot(x2 - x3, y2 - y3)
            
            if distance_right < 40:
                cv2.circle(frame, (x3, y3), 15, (0, 0, 255), cv2.FILLED) # Red feedback
                if mouse:
                    mouse.click('right')
                    time.sleep(0.1)

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