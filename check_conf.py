"""Проверка порога уверенности (CONF) для app.py.

Запуск:
    .venv\\Scripts\\python check_conf.py [картинка]

Если картинка не указана, берётся первый файл из screenshots/.
Скрипт прогоняет модель при разных порогах и показывает, какие классы
и с какой уверенностью находятся. Так видно, как CONF влияет на детекции.
"""
import sys
import glob
import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = ["game", "upgrade", "reclama"]
IMGSZ = 640
NMS_THRESHOLD = 0.45

session = ort.InferenceSession("bestV1.onnx", providers=["CPUExecutionProvider"])
IN = session.get_inputs()[0].name
OUT = session.get_outputs()[0].name


def letterbox(frame, imgsz):
    h, w = frame.shape[:2]
    scale = imgsz / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    r = cv2.resize(frame, (nw, nh))
    c = np.full((imgsz, imgsz, 3), 114, np.uint8)
    c[:nh, :nw] = r
    return c


def raw_scores(frame, conf=0.0):
    img = letterbox(frame, IMGSZ)
    inp = img.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None]
    out = session.run([OUT], {IN: inp})[0]
    p = out[0].T
    cs = p[:, 4:]
    cid = np.argmax(cs, 1)
    sc = cs[np.arange(len(cid)), cid]
    m = sc >= conf
    return cid[m], sc[m]


def detect_at(frame, conf):
    cid, sc = raw_scores(frame, conf)
    bx = None  # boxes not needed for this check
    img = letterbox(frame, IMGSZ)
    inp = img.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None]
    out = session.run([OUT], {IN: inp})[0]
    p = out[0].T
    boxes, cs = p[:, :4], p[:, 4:]
    c = np.argmax(cs, 1)
    s = cs[np.arange(len(c)), c]
    m = s >= conf
    boxes, s, c = boxes[m], s[m], c[m]
    if len(s) == 0:
        return {}
    keep = np.array(cv2.dnn.NMSBoxes(boxes.tolist(), s.tolist(), conf, NMS_THRESHOLD)).flatten().tolist()
    res = {}
    for i in keep:
        res.setdefault(CLASS_NAMES[c[i]], []).append(round(float(s[i]), 2))
    return res


path = sys.argv[1] if len(sys.argv) > 1 else None
if path is None:
    files = sorted(glob.glob("screenshots/*.png")) + sorted(glob.glob("images/val/*.jpg"))
    if not files:
        print("Нет картинок для проверки. Передайте путь аргументом.")
        raise SystemExit(1)
    path = files[0]

print("Картинка:", path)
bgr = cv2.imread(path)
if bgr is None:
    print("Не удалось прочитать файл.")
    raise SystemExit(1)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

print("\nДетекции при разных порогах уверенности:")
for conf in [0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.15, 0.05]:
    print(f"  CONF={conf:<5} -> {detect_at(rgb, conf)}")
