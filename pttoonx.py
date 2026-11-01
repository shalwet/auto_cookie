from ultralytics import YOLO

model = YOLO("bestV1.pt")

# Укажите реальный файл из вашего датасета
test_image = "images/train/screenshot_2026-10-29_231703_aug0_1.jpg"

results = model(test_image, save=True, conf=0.25)
print(f"Результат в: runs/detect/predict/")