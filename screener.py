import pyautogui
import time
from datetime import datetime
import os 

save_dir = 'screenshorts'
os.makedirs(save_dir, exist_ok=True)

while True:
      screenshot = pyautogui.screenshot()

      timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
      filename = f"{save_dir}/screenshot_{timestamp}.png"

      screenshot.save(filename)
      time.sleep(5)