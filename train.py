from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="cpu",           # <-- ИСПРАВЛЕНО: было device=0
        workers=4,
        project="runs/train",
        name="my_model",
        exist_ok=True,
        patience=10,
        save=True,
        verbose=True,
    )
    
    print("Обучение завершено. Лучшие веса: runs/train/my_model/weights/best.pt")