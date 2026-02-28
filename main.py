# main application file
import cv2
import time
import math
import numpy as np
from hand_detector import HandDetector
from utils.drawing_utils import draw_hand_points, draw_bounding_box, draw_hand_skeleton
from mouse_controller import MouseController
from config import (
    CAMERA_INDEX, CAM_WIDTH, CAM_HEIGHT,
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
    SMOOTHING_ALPHA, FRAME_REDUCTION,
    PINCH_DISTANCE, SCROLL_JITTER_THRESHOLD, SCROLL_SPEED_MULTIPLIER,
    CLICK_COOLDOWN, FINGER_CIRCLE_RADIUS, WINDOW_TITLE,
)

# Initialize MouseController
if MouseController:
    mouse = MouseController(alpha=SMOOTHING_ALPHA)
else:
    mouse = None

# Initialize hand detector
detector = HandDetector(MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE)

# Initialize webcam
cap = cv2.VideoCapture(CAMERA_INDEX)

# Set frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# Check if camera opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Runtime state variables (NOT config — these change every frame)
prev_time = 0
prev_y1 = 0
mode = "mouse"

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
        hand = hands_data[0]
        positions = hand["positions"]
        hand_landmarks = hand["landmarks"]
        
        if mode == "mouse":
            # Draw Visuals
            draw_hand_points(frame, positions)
            draw_hand_skeleton(frame, hand_landmarks, detector.mp_hands, detector.mp_drawing)
        
            if positions:
                # EXTRACT LANDMARKS
                x1, y1 = positions[8][1], positions[8][2]   # Index Finger Tip
                x2, y2 = positions[4][1], positions[4][2]   # Thumb Tip
                x3, y3 = positions[12][1], positions[12][2] # Middle Finger Tip
                x4, y4 = positions[16][1], positions[16][2] # Ring Finger Tip
                
                # CHECK SCROLL GESTURE (Thumb + Ring) first
                dist_scroll = math.hypot(x2 - x4, y2 - y4)
                
                if dist_scroll < PINCH_DISTANCE:
                    # SCROLL MODE
                    cv2.circle(frame, (x4, y4), FINGER_CIRCLE_RADIUS, (255, 255, 0), cv2.FILLED)
                    
                    if prev_y1 == 0: prev_y1 = y1
                    delta_y = y1 - prev_y1
                    
                    if abs(delta_y) > SCROLL_JITTER_THRESHOLD:
                        scroll_amount = int(-delta_y * SCROLL_SPEED_MULTIPLIER)
                        if mouse:
                            mouse.scroll(scroll_amount)
                    
                else:
                    # NORMAL MOUSE MODE (Move + Click)
                    cv2.circle(frame, (x1, y1), FINGER_CIRCLE_RADIUS, (255, 0, 255), cv2.FILLED)
                    cv2.circle(frame, (x2, y2), FINGER_CIRCLE_RADIUS, (255, 0, 255), cv2.FILLED)

                    # Move Mouse
                    if mouse:
                        x3_screen = np.interp(x1, (FRAME_REDUCTION, CAM_WIDTH - FRAME_REDUCTION), (0, mouse.screen_w))
                        y3_screen = np.interp(y1, (FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION), (0, mouse.screen_h))
                        mouse.move(x3_screen, y3_screen)

                    # Left Click (Thumb + Index)
                    distance = math.hypot(x2 - x1, y2 - y1)
                    if distance < PINCH_DISTANCE:
                        cv2.circle(frame, (x1, y1), FINGER_CIRCLE_RADIUS, (0, 255, 0), cv2.FILLED)
                        if mouse:
                            mouse.click('left')
                            time.sleep(CLICK_COOLDOWN)

                    # Right Click (Thumb + Middle)
                    distance_right = math.hypot(x2 - x3, y2 - y3)
                    if distance_right < PINCH_DISTANCE:
                        cv2.circle(frame, (x3, y3), FINGER_CIRCLE_RADIUS, (0, 0, 255), cv2.FILLED)
                        if mouse:
                            mouse.click('right')
                            time.sleep(CLICK_COOLDOWN)

            # Update previous y1 for next frame
            prev_y1 = y1
        
        elif mode == "sign_language":
            cv2.putText(frame, "Sign Language Mode - classifier not loaded yet", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            draw_hand_points(frame, positions)
            draw_hand_skeleton(frame, hand_landmarks, detector.mp_hands, detector.mp_drawing)
    
    # Calculate fps
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # Display fps and mode
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Mode: {mode}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow(WINDOW_TITLE, frame)

    # Handle keypresses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        mode = "sign_language" if mode == "mouse" else "mouse"
    if key == 27 or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()