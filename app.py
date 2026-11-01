import os
import cv2
import numpy as np
import pyautogui
import time
import keyboard
import threading

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
game_pos = None                 # последняя позиция game (в координатах экрана)
click_lock = threading.Lock()   # мышь — только по одному

def toggle():
    global running
    running = not running
    print(">>>", "ВКЛ" if running else "ВЫКЛ")

keyboard.add_hotkey("space", toggle)

# ===== Настройка уверенности (порог детекции) =====
CONF_FILE = "conf.txt"


def load_conf(default=0.25):
    """Читает CONF из conf.txt (одно число). Если файла нет — берёт из config.yaml, иначе default."""
    try:
        with open(CONF_FILE, "r", encoding="utf-8") as f:
            return float(f.read().strip())
    except Exception:
        pass
    try:
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return float(cfg.get("confidence_threshold", default))
    except Exception:
        return default


def save_conf(value):
    try:
        with open(CONF_FILE, "w", encoding="utf-8") as f:
            f.write(f"{value:.2f}")
    except Exception as e:
        print("Не удалось сохранить conf.txt:", e)


CONF = load_conf()
CONF_STEP = 0.05          # шаг изменения уверенности по хоткеям
CONF_MIN = 0.05
CONF_MAX = 0.95


def conf_up():
    global CONF
    CONF = min(CONF_MAX, round(CONF + CONF_STEP, 2))
    save_conf(CONF)
    print(f">>> уверенность (CONF) = {CONF:.2f}")


def conf_down():
    global CONF
    CONF = max(CONF_MIN, round(CONF - CONF_STEP, 2))
    save_conf(CONF)
    print(f">>> уверенность (CONF) = {CONF:.2f}")


# '+'/'-' (и '='/'_' как запасные) — поднять/опустить порог уверенности на ходу
keyboard.add_hotkey("+", conf_up)
keyboard.add_hotkey("=", conf_up)
keyboard.add_hotkey("-", conf_down)
keyboard.add_hotkey("_", conf_down)

NMS_THRESHOLD = 0.45
SLEEP = 0.1

# Печатать отладку по найденным классам каждый кадр
DEBUG_DETECTIONS = True

# Шаг вертикального прохода по боксу upgrade (в пикселях)
UPGRADE_STEP = 5

# Смещение курсора по upgrade вправо (в пикселях)
UPGRADE_X_OFFSET = 20

# Пауза между кликами по game в фоновом потоке (сек) — меньше = чаще
GAME_CLICK_INTERVAL = 0.02

# Возвращать курсор на место после клика по game (чтобы курсор не «уезжал»)
GAME_RESTORE_CURSOR = True


def game_clicker_loop():
    """Кликает по game НЕПРЕРЫВНО, без остановки, всё время пока включён кликер.

    Если GAME_RESTORE_CURSOR=True — после клика возвращает курсор на прежнее
    место, чтобы мышь не «уезжала» к объекту game (обычно он справа).
    """
    while True:
        pos = game_pos
        if running and pos is not None:
            with click_lock:
                if GAME_RESTORE_CURSOR:
                    prev = pyautogui.position()
                    pyautogui.click(pos[0], pos[1])
                    pyautogui.moveTo(prev[0], prev[1], duration=0)
                else:
                    pyautogui.click(pos[0], pos[1])
            time.sleep(GAME_CLICK_INTERVAL)
        else:
            time.sleep(0.005)


# Фоновый поток непрерывных кликов по game
threading.Thread(target=game_clicker_loop, daemon=True).start()

print("ПРОБЕЛ — старт/стоп кликера. Esc — выход.")
print(f"Настройка уверенности: '+' повысить, '-' понизить (текущая CONF = {CONF:.2f}, сохраняется в conf.txt)")
print("Целевые классы: game, upgrade, reclama")


def letterbox(frame, imgsz):
    """Ресайз с сохранением пропорций и паддинг до квадрата imgsz (top-left)."""
    h, w = frame.shape[:2]
    scale = imgsz / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas, scale


def nms(boxes_xywh, scores, score_threshold, nms_threshold):
    """Индексы оставшихся боксов (cv2.dnn.NMSBoxes в формате xywh)."""
    if len(boxes_xywh) == 0:
        return []
    idxs = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), scores.tolist(), score_threshold, nms_threshold
    )
    return np.array(idxs).flatten().tolist()


def nms_per_class(boxes_xywh, scores, class_ids, score_threshold, nms_threshold):
    """NMS ОТДЕЛЬНО по каждому классу.

    Обычный NMS подавляет любые пересекающиеся боксы, поэтому перекрывающийся
    высокоуверенный `game` мог «съедать» `upgrade`/`reclama`.
    """
    keep = []
    for c in np.unique(class_ids):
        idx = np.where(class_ids == c)[0]
        if idx.size == 0:
            continue
        kept = nms(boxes_xywh[idx], scores[idx], score_threshold, nms_threshold)
        keep.extend(idx[j] for j in kept)
    return keep


def infer(frame):
    """Инференс ONNX. Возвращает список (name, conf, x1, y1, x2, y2) в координатах экрана."""
    img, scale = letterbox(frame, IMGSZ)

    # frame уже в RGB (pyautogui.screenshot), ONNX-модель ожидает RGB.
    inp = img.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]     # HWC -> 1CHW

    out = session.run([output_name], {input_name: inp})[0]  # (1, 4+nc, N)

    preds = np.squeeze(out, axis=0).T        # (N, 4+nc)
    boxes_xywh = preds[:, :4]                # cx, cy, w, h (в координатах letterbox)
    class_scores = preds[:, 4:]              # nc столбцов

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]

    mask = scores >= CONF
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(scores) == 0:
        return []

    keep = nms_per_class(boxes_xywh, scores, class_ids, CONF, NMS_THRESHOLD)

    results = []
    for i in keep:
        name = CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else str(class_ids[i])

        cx, cy, bw, bh = boxes_xywh[i]
        x1 = int((cx - bw / 2) / scale)
        y1 = int((cy - bh / 2) / scale)
        x2 = int((cx + bw / 2) / scale)
        y2 = int((cy + bh / 2) / scale)

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

    if DEBUG_DETECTIONS:
        by_class = {}
        for nm, cf, *_ in detections:
            by_class.setdefault(nm, []).append(round(cf, 2))
        print(f"[{time.strftime('%H:%M:%S')}] detections: {by_class}")

    # Собираем ВСЕ целевые детекции
    targets = []
    game_pos = None  # сброс: если game не найден, поток перестанет кликать
    for name, conf, x1, y1, x2, y2 in detections:
        if name not in ("game", "upgrade", "reclama"):
            continue

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        targets.append((name, conf, x1, y1, x2, y2))

    # Приоритет: сначала game, затем upgrade, затем reclama
    priority = {"game": 0, "upgrade": 1, "reclama": 2}
    targets.sort(key=lambda t: (priority[t[0]], -t[1]))

    # Позиция game (кликается фоновым потоком)
    for name, conf, x1, y1, x2, y2 in targets:
        if name == "game":
            game_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
            break

    # Проход по upgrade и клик по reclama. game НЕ останавливается —
    # он продолжает щёлкать в фоне, а click_lock аккуратно чередует клики.
    for name, conf, x1, y1, x2, y2 in targets:
        if name == "game":
            continue
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if name == "upgrade":
            cx = min(cx + UPGRADE_X_OFFSET, w - 1)
            print(f"[{time.strftime('%H:%M:%S')}] upgrade ({conf:.2f}) бокс ({x1},{y1})-({x2},{y2})")
            y = y1
            while y <= y2:
                with click_lock:
                    pyautogui.moveTo(cx, y, duration=0.02)
                    pyautogui.click()
                time.sleep(0.01)
                y += UPGRADE_STEP
        elif name == "reclama":
            print(f"[{time.strftime('%H:%M:%S')}] reclama ({conf:.2f}) в ({cx}, {cy})")
            with click_lock:
                pyautogui.moveTo(cx, cy, duration=0.05)
                pyautogui.click()
            time.sleep(0.05)

    time.sleep(SLEEP)
