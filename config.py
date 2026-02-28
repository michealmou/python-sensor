"""
config.py — Centralized configuration constants for SignSense.

All tunable parameters live here. Do NOT put runtime state
variables (counters, accumulators, etc.) in this file.
"""

# ── Camera ────────────────────────────────────────────
CAMERA_INDEX = 0              # Default webcam index
CAM_WIDTH = 1280              # Capture width
CAM_HEIGHT = 720              # Capture height

# ── Hand Detection (MediaPipe) ────────────────────────
MAX_HANDS = 2                 # Max simultaneous hands
DETECTION_CONFIDENCE = 0.7    # Min detection confidence
TRACKING_CONFIDENCE = 0.7     # Min tracking confidence

# ── Mouse Controller ─────────────────────────────────
SMOOTHING_ALPHA = 0.3         # Exponential smoothing (0.2–0.35)
FRAME_REDUCTION = 150         # Edge padding for coordinate mapping

# ── Gesture Thresholds ────────────────────────────────
PINCH_DISTANCE = 40           # Pixels to register a pinch
SCROLL_JITTER_THRESHOLD = 5   # Min Y-delta to trigger scroll
SCROLL_SPEED_MULTIPLIER = 1.5 # Scroll speed factor
CLICK_COOLDOWN = 0.1          # Seconds to wait after a click

# ── Drawing ───────────────────────────────────────────
FINGER_CIRCLE_RADIUS = 15     # Radius for fingertip circles

# ── Display ───────────────────────────────────────────
WINDOW_TITLE = "Hand Tracking"