import cv2
import numpy as np
import pyautogui
import time
import keyboard
import threading

from ultralytics import YOLO

model = YOLO("bestV1.pt")

pyautogui.PAUSE = 0.02
pyautogui.FAILSAFE = True

running = False
game_pos = None          # текущая позиция game (обновляется из основного цикла)
mouse_lock = threading.Lock()

def toggle():
    global running
    running = not running
    print(">>>", "ВКЛ" if running else "ВЫКЛ")

keyboard.add_hotkey("space", toggle)

CONF = 0.5
SLEEP = 0.1

# Шаг вертикального прохода по боксу upgrade (в пикселях)
UPGRADE_STEP = 5

# Смещение курсора по upgrade вправо (в пикселях)
UPGRADE_X_OFFSET = 20

# Как часто отдельный поток кликает по game (сек) — меньше = чаще
GAME_CLICK_INTERVAL = 0.03

def game_clicker_loop():
    """Кликает по game ПАРАЛЛЕЛЬНО с проходом по upgrade."""
    while True:
        pos = game_pos
        if running and pos is not None:
            with mouse_lock:
                pyautogui.click(pos[0], pos[1])
            time.sleep(GAME_CLICK_INTERVAL)
        else:
            time.sleep(0.01)

# Отдельный поток для частых кликов по game
threading.Thread(target=game_clicker_loop, daemon=True).start()

# >>> РАЗРЕШЕНИЕ ОБУЧЕНИЯ (проверьте: model.overrides.get('imgsz'))
IMGSZ = 640

print("ПРОБЕЛ — старт/стоп кликера. Esc — выход.")
print("Целевые классы: game, upgrade, reclama")

while True:
    if not running:
        time.sleep(0.05)
        continue

    # Скриншот экрана
    screen = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

    # >>> РЕСАЙЗ ПОД IMGSZ + ПАДДИНГ (letterbox)
    h, w = frame.shape[:2]
    scale = IMGSZ / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    # Инференс на 640x640
    results = model(canvas, conf=CONF, verbose=False)

    # Собираем ВСЕ целевые детекции (не только первую!)
    targets = []
    game_pos = None  # сброс: если game не найден, поток перестанет кликать
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            if name not in ("game", "upgrade", "reclama"):
                continue

            conf = float(box.conf[0])

            # >>> ОБРАТНЫЙ МАСШТАБ БОКСОВ В ЭКРАННЫЕ КООРДИНАТЫ
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)

            # Страховка: не выходим за пределы экрана
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            targets.append((name, conf, x1, y1, x2, y2))

    # Приоритет: сначала game, затем upgrade, затем reclama (по убыванию conf внутри класса)
    priority = {"game": 0, "upgrade": 1, "reclama": 2}
    targets.sort(key=lambda t: (priority[t[0]], -t[1]))

    for name, conf, x1, y1, x2, y2 in targets:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if name == "game":
            # Позицию game отдаём отдельному потоку — он кликает часто и параллельно
            game_pos = (cx, cy)
        elif name == "upgrade":
            # Кликаем по всему боксу сверху вниз, чуть правее центра
            cx = min(cx + UPGRADE_X_OFFSET, w - 1)
            print(f"[{time.strftime('%H:%M:%S')}] upgrade ({conf:.2f}) бокс ({x1},{y1})-({x2},{y2})")
            y = y1
            while y <= y2:
                with mouse_lock:
                    pyautogui.moveTo(cx, y, duration=0.02)
                    time.sleep(0.01)
                    pyautogui.click()
                time.sleep(0.02)
                y += UPGRADE_STEP
        elif name == "reclama":
            print(f"[{time.strftime('%H:%M:%S')}] reclama ({conf:.2f}) в ({cx}, {cy})")
            with mouse_lock:
                pyautogui.moveTo(cx, cy, duration=0.05)
                time.sleep(0.02)
                pyautogui.click()
            time.sleep(0.05)

    time.sleep(SLEEP)