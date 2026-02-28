# main application file
import cv2
import time
import math
import numpy as np
from hand_detector import HandDetector
from utils.drawing_utils import draw_hand_points, draw_bounding_box, draw_hand_skeleton
from mouse_controller import MouseController
from config import max_hands, detection_confidence, tracking_confidence, alpha, frameR, CAM_WIDTH, CAM_HEIGHT   

# Initialize MouseController
if MouseController:
    mouse = MouseController(alpha)

else:
    mouse = None

# Frame Reduction (padding from the edge of the webcam view)
frameR = 150

# Initialize hand detector
detector = HandDetector(max_hands, detection_confidence, tracking_confidence)

# Initialize webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Set frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# Check if camera opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Initialize variables
prev_time = 0
prev_y1 = 0

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
            # 1. GESTURE LOGIC
            # EXTRACT LANDMARKS
            # Index Finger Tip (8)
            x1, y1 = positions[8][1], positions[8][2]
            # Thumb Tip (4)
            x2, y2 = positions[4][1], positions[4][2]
            # Middle Finger Tip (12)
            x3, y3 = positions[12][1], positions[12][2]
            # Ring Finger Tip (16)
            x4, y4 = positions[16][1], positions[16][2]
            
            # CHECK SCROLL GESTURE (Thumb + Ring) first
            dist_scroll = math.hypot(x2 - x4, y2 - y4)
            
            if dist_scroll < 40:
                # SCROLL MODE ACTIVATED
                cv2.circle(frame, (x4, y4), 15, (255, 255, 0), cv2.FILLED) # Cyan Visual
                
                # Calculate vertical movement from previous Index Finger position (y1)
                # Note: We need a stored position, but for simplicity in this loop 
                # we can use the relative movement if we had prev_y. 
                # Since we don't track prev_y globally for logic outside mouse_controller,
                # let's use the mouse_controller's own stored position or a simple diff approach?
                
                # Actually, simpler approach: 
                # Map the hand's Y position to a "Scroll Speed"?
                # No, "take fingers up" implies movement.
                
                # Let's track prev_y1 in the loop.
                # Initialize it at top of loop if 0.
                if prev_y1 == 0: prev_y1 = y1
                
                delta_y = y1 - prev_y1
                
                if abs(delta_y) > 5: # Threshold to reduce jitter
                    # Inverted: Moving hand UP (negative delta) -> Scroll UP (positive)
                    # Moving hand DOWN (positive delta) -> Scroll DOWN (negative)
                    scroll_amount = int(-delta_y * 1.5) # 1.5 multiplier for speed
                    if mouse:
                        mouse.scroll(scroll_amount)
                
            else:
                # NORMAL MOUSE MODE (Move + Click)
                
                # Draw circles for key fingers
                cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

                # Move Mouse
                if mouse:
                    x3_screen = np.interp(x1, (frameR, 1280 - frameR), (0, mouse.screen_w))
                    y3_screen = np.interp(y1, (frameR, 720 - frameR), (0, mouse.screen_h))
                    mouse.move(x3_screen, y3_screen)

                # Left Click (Thumb + Index)
                distance = math.hypot(x2 - x1, y2 - y1)
                if distance < 40:
                    cv2.circle(frame, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    if mouse:
                        mouse.click('left')
                        time.sleep(0.1)

                # Right Click (Thumb + Middle)
                distance_right = math.hypot(x2 - x3, y2 - y3)
                if distance_right < 40:
                    cv2.circle(frame, (x3, y3), 15, (0, 0, 255), cv2.FILLED) # Red
                    if mouse:
                        mouse.click('right')
                        time.sleep(0.1)

            # Update previous y1 for next frame
            prev_y1 = y1

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