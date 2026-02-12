import time
import pyautogui

# Make pyautogui fast (VERY important)
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True


class MouseController:
    def __init__(self, alpha=0.25, move_interval=0.01, dead_zone=3):
        """
        Mouse controller with smoothing, safety, and throttling.

        alpha         -> smoothing strength (0.2–0.35 is good)
        move_interval -> max mouse update rate (seconds)
        dead_zone     -> ignore tiny movements (pixels)
        """

        # Screen size
        self.screen_w, self.screen_h = pyautogui.size()

        # Current mouse position
        self.current_x, self.current_y = pyautogui.position()

        # Smoothing + stability settings
        self.alpha = alpha
        self.dead_zone = dead_zone

        # Timing (prevents FPS drops)
        self.move_interval = move_interval
        self.last_move_time = 0

    def move(self, x, y):
        """Move mouse smoothly to (x, y)"""

        # Rate limit mouse updates
        now = time.time()
        if now - self.last_move_time < self.move_interval:
            return
        self.last_move_time = now

        # Clamp target to screen bounds
        x = max(0, min(self.screen_w - 1, x))
        y = max(0, min(self.screen_h - 1, y))

        # Ignore tiny jitter
        if abs(x - self.current_x) < self.dead_zone and abs(y - self.current_y) < self.dead_zone:
            return

        # Exponential smoothing (stable + responsive)
        self.current_x = self.current_x * (1 - self.alpha) + x * self.alpha
        self.current_y = self.current_y * (1 - self.alpha) + y * self.alpha

        # Move real OS cursor
        pyautogui.moveTo(int(self.current_x), int(self.current_y))

    def click(self, button="left"):
        pyautogui.click(button=button)

    def scroll(self, dy):
        pyautogui.scroll(dy)


# =========================
# TEST CODE (safe to keep)
# =========================
if __name__ == "__main__":
    mouse = MouseController(alpha=0.3)

    square = [
        (200, 200),
        (600, 200),
        (600, 600),
        (200, 600),
        (200, 200),
    ]

    print("Testing mouse controller...")
    print("Move mouse to TOP-LEFT corner to abort")

    for x, y in square:
        for _ in range(40):
            mouse.move(x, y)
            time.sleep(0.01)
        time.sleep(0.3)

    print("Testing complete")
