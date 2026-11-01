import cv2
import numpy as np
import pyautogui
import time
import keyboard

import onnxruntime as ort

# ===== ONNX-модель =====
ONNX_PATH = "bestV1.onnx"
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Имена классов (совпадают с class.txt / data.yaml)
CLASS_NAMES = ["game", "upgrade", "reclama"]

# >>> РАЗРЕШЕНИЕ ОБУЧЕНИЯ / ЭКСПОРТА ONNX (обычно 640)
IMGSZ = 640

pyautogui.PAUSE = 0.02
pyautogui.FAILSAFE = True

running = False
game_pos = None          # текущая позиция game (обновляется из основного цикла)

def toggle():
    global running
    running = not running
    print(">>>", "ВКЛ" if running else "ВЫКЛ")

keyboard.add_hotkey("space", toggle)

CONF = 0.5
NMS_THRESHOLD = 0.45
SLEEP = 0.1

# Шаг вертикального прохода по боксу upgrade (в пикселях)
UPGRADE_STEP = 5

# Смещение курсора по upgrade вправо (в пикселях)
UPGRADE_X_OFFSET = 20

# Пауза между кликами по game (сек) — меньше = чаще
GAME_CLICK_INTERVAL = 0.03

# Сколько раз подряд жать game, когда upgrade/reclama на экране нет
GAME_BURST = 20

print("ПРОБЕЛ — старт/стоп кликера. Esc — выход.")
print("Целевые классы: game, upgrade, reclama")


def letterbox(frame, imgsz):
    """Ресайз с сохранением пропорций и паддинг до квадрата imgsz (top-left, как раньше)."""
    h, w = frame.shape[:2]
    scale = imgsz / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas, scale, new_w, new_h


def nms(boxes_xywh, scores, score_threshold, nms_threshold):
    """Возвращает индексы оставшихся боксов (cv2.dnn.NMSBoxes в формате xywh)."""
    if len(boxes_xywh) == 0:
        return []
    idxs = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), scores.tolist(), score_threshold, nms_threshold
    )
    return np.array(idxs).flatten().tolist()


def infer(frame):
    """Инференс через ONNX Runtime. Возвращает список (name, conf, x1, y1, x2, y2) в координатах экрана."""
    img, scale, new_w, new_h = letterbox(frame, IMGSZ)

    # frame уже в RGB (pyautogui.screenshot), ONNX-модель ожидает RGB.
    # NCHW + нормализация 0..1 (без перестановки каналов!)
    inp = img.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]     # HWC -> 1CHW

    out = session.run([output_name], {input_name: inp})[0]  # (1, 4+nc, N)

    preds = np.squeeze(out, axis=0)          # (4+nc, N)
    preds = preds.T                          # (N, 4+nc)
    boxes_xywh = preds[:, :4]                # cx, cy, w, h
    class_scores = preds[:, 4:]              # nc столбцов

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]

    mask = scores >= CONF
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(scores) == 0:
        return []

    keep = nms(boxes_xywh, scores, CONF, NMS_THRESHOLD)

    results = []
    for i in keep:
        name = CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else str(class_ids[i])

        cx, cy, bw, bh = boxes_xywh[i]
        # cxcywh -> xyxy в координатах letterbox
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # убираем масштаб letterbox -> экранные координаты
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)

        results.append((name, float(scores[i]), x1, y1, x2, y2))

    return results


while True:
    if not running:
        time.sleep(0.05)
        continue

    # Скриншот экрана
    screen = pyautogui.screenshot()
    frame = np.array(screen)                # RGB
    h, w = frame.shape[:2]

    # Инференс ONNX
    detections = infer(frame)

    # Собираем ВСЕ целевые детекции (не только первую!)
    targets = []
    game_pos = None  # сброс: если game не найден, поток перестанет кликать
    for name, conf, x1, y1, x2, y2 in detections:
        if name not in ("game", "upgrade", "reclama"):
            continue

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
            # Кликаем по всему боксу сверху вниз, чуть правее центра.
            # Между шагами жмём game, чтобы он кликался НЕ прекращаясь.
            cx = min(cx + UPGRADE_X_OFFSET, w - 1)
            print(f"[{time.strftime('%H:%M:%S')}] upgrade ({conf:.2f}) бокс ({x1},{y1})-({x2},{y2})")
            y = y1
            while y <= y2:
                pyautogui.moveTo(cx, y, duration=0.02)
                pyautogui.click()
                time.sleep(0.01)
                if game_pos is not None:
                    pyautogui.click(game_pos[0], game_pos[1])
                y += UPGRADE_STEP
        elif name == "reclama":
            print(f"[{time.strftime('%H:%M:%S')}] reclama ({conf:.2f}) в ({cx}, {cy})")
            pyautogui.moveTo(cx, cy, duration=0.05)
            time.sleep(0.02)
            pyautogui.click()
            time.sleep(0.05)
            if game_pos is not None:
                pyautogui.click(game_pos[0], game_pos[1])

    # Game кликаем ВСЕГДА и по многу раз — «без остановки»
    if game_pos is not None:
        for _ in range(GAME_BURST):
            pyautogui.click(game_pos[0], game_pos[1])
            time.sleep(GAME_CLICK_INTERVAL)

    time.sleep(SLEEP)
