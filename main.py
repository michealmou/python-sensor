"""
Hand Detection Project
Entry point - Run this file to start the hand detection
"""

import cv2
from hand_detector import HandDetector


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize hand detector
    detector = HandDetector(max_hands=2, detection_confidence=0.7)

    print("Hand Detection Started! Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Detect hands and draw landmarks
        frame, hands = detector.find_hands(frame, draw=True)

        # Display hand count
        if hands:
            cv2.putText(frame, f"Hands detected: {len(hands)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Hand Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
