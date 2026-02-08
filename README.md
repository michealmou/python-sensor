# Hand Detection Project 🖐️

Real-time hand detection and tracking using OpenCV and MediaPipe.

![Demo](assets/demo.png)

## Features

- 🎯 Real-time hand detection (up to 2 hands)
- 📍 21-point hand landmark tracking
- ✋ Finger counting
- 🏷️ Left/Right hand identification
- 📦 Bounding box visualization

## Installation

1. Clone or download this project
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python main.py
```

- Press **Q** to quit the application

## Project Structure

```
hand-detection-project/
├── main.py              # Entry point (run this)
├── hand_detector.py     # Hand detection logic
├── requirements.txt     # Libraries list
├── README.md            # Project explanation
├── utils/
│   └── drawing_utils.py # Drawing functions
└── assets/
    └── demo.png         # Screenshots / demo images
```

## How It Works

1. **Capture**: Reads frames from your webcam
2. **Process**: MediaPipe detects hands and extracts 21 landmarks per hand
3. **Draw**: Custom utilities visualize landmarks and connections
4. **Display**: OpenCV shows the annotated video feed

## API Reference

### HandDetector Class

```python
from hand_detector import HandDetector

detector = HandDetector(max_hands=2, detection_confidence=0.7)
frame, hands = detector.find_hands(frame, draw=True)
```

#### Methods

| Method | Description |
|--------|-------------|
| `find_hands(frame, draw)` | Detect hands and return landmark data |
| `get_finger_tips(hand)` | Get fingertip positions |
| `count_fingers(hand)` | Count raised fingers (0-5) |

## License

MIT License - Feel free to use and modify!