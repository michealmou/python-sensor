import pyautogui


class MouseController:
    def __init__(self, smooth_factor=5):
        # Get screen size
        self.screen_w, self.screen_h = pyautogui.size()

        # Store smoothing factor
        self.smooth = smooth_factor

        # Start from current mouse position
        self.current_x, self.current_y = pyautogui.position()

        # Safety switch
        pyautogui.FAILSAFE = True

    def move(self, x, y):
        # Clamp to screen bounds
        x = max(0, min(self.screen_w - 1, x))
        y = max(0, min(self.screen_h - 1, y))

        # Smooth movement
        self.current_x += (x - self.current_x) / self.smooth
        self.current_y += (y - self.current_y) / self.smooth

        # Move OS mouse
        pyautogui.moveTo(int(self.current_x), int(self.current_y))

    def click(self, button="left"):
        pyautogui.click(button=button)

    def scroll(self, dy):
        pyautogui.scroll(dy)


# =========================
# TEST CODE
# =========================
if __name__ == "__main__":
    import time

    mouse = MouseController(smooth_factor=6)

    square = [
        (200, 200),
        (600, 200),
        (600, 600),
        (200, 600),
        (200, 200),
    ]

    print("Testing mouse controller... Move mouse to top-left to stop.")

    for x, y in square:
        for _ in range(40):
            mouse.move(x, y)
            time.sleep(0.01)
        time.sleep(0.3)

    print("Testing complete")
