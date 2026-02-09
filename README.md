# 🤟 SignSense

**Real-time Sign Language Interpreter using Computer Vision**

A Python-based hand tracking and gesture recognition system that uses MediaPipe and OpenCV to detect hand landmarks, identify individual finger positions, and interpret sign language in real-time.

---

## ✨ Features

- 🖐️ **Real-time Hand Detection** — Tracks up to 2 hands simultaneously
- 🎯 **21-Point Landmark Tracking** — Precise finger joint detection
- 🔄 **Live Webcam Feed** — Mirror-mode display with FPS counter
- 🏷️ **Hand Classification** — Distinguishes between left and right hands
- 📦 **Modular Architecture** — Clean separation of detection and drawing utilities

## 🛠️ Tech Stack

- **Python 3.9–3.11** (recommended for MediaPipe compatibility)
- **OpenCV** — Video capture and image processing
- **MediaPipe** — Hand landmark detection

## 📁 Project Structure

```
python-sensor/
├── main.py              # Main application loop
├── hand_detector.py     # HandDetector class for detection logic
├── utils/
│   ├── __init__.py
│   └── drawing_utils.py # Drawing helper functions
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python mediapipe
```

### Run

```bash
python main.py
```

Press **ESC** or click the **X** button to exit.

## 🗺️ Roadmap

- [ ] Finger state detection (open/closed)
- [ ] Finger counting
- [ ] Basic sign language gesture recognition
- [ ] ASL alphabet interpretation

---

Made with ❤️ and Python