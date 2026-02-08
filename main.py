import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Set frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# chech if camera opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()


# Initialize variables
prev_time = 0

# open webcam & reading frames
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hands
    results = hands.process(rgb_frame)
    
    # Check if hands detected
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get hand type (Left or Right)
            hand_type = results.multi_handedness[hand_idx].classification[0].label

            # Get frame dimensions
            h, w, _ = frame.shape
            
            # Collect landmark coordinates for bounding box
            x_coords = []
            y_coords = []
            landmarks = []
            
            for lm in hand_landmarks.landmark:
                px, py = int(lm.x * w), int(lm.y * h)
                landmarks.append((px, py))
                x_coords.append(px)
                y_coords.append(py)
            
            # Draw hand landmarks (21 points)
            for idx, (px, py) in enumerate(landmarks):
                # Fingertips get larger circles
                radius = 8 if idx in [4, 8, 12, 16, 20] else 5
                cv2.circle(frame, (px, py), radius, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, (px, py), radius, (0, 0, 0), 1)  # Black outline
            
            # Draw connections between landmarks
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Calculate bounding box
            x_min, x_max = min(x_coords) - 20, max(x_coords) + 20
            y_min, y_max = min(y_coords) - 20, max(y_coords) + 20
            
            # Draw bounding box
            box_color = (255, 0, 0) if hand_type == "Left" else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
            
            # Draw text label ("Left Hand" / "Right Hand")
            label = f"{hand_type} Hand"
            cv2.rectangle(frame, (x_min, y_min - 30), (x_min + 120, y_min), box_color, cv2.FILLED)
            cv2.putText(frame, label, (x_min + 5, y_min - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # calculate fps
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # display fps
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # show the frame
    cv2.imshow('Hand Tracking', frame)

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera
cap.release()
cv2.destroyAllWindows()