# split_dataset.py — разделяет фото И разметку на train/val синхронно
# Использование: python split_dataset.py

import os
import random
import shutil
from pathlib import Path

# === НАСТРОЙКИ ===
SOURCE_IMAGES = "screenshots_augmented"   # папка с фото
SOURCE_LABELS = "labels"                  # папка с .txt разметкой
DEST = "itog"                             # куда положить результат
TRAIN_RATIO = 0.8
MODE = "copy"                             # "copy" или "move"
SEED = 42
# =================

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

def main():
    random.seed(SEED)
    
    src_img = Path(SOURCE_IMAGES)
    src_lbl = Path(SOURCE_LABELS)
    dst = Path(DEST)
    
    if not src_img.exists():
        print(f"❌ Нет папки с фото: {src_img}")
        return
    if not src_lbl.exists():
        print(f"❌ Нет папки с разметкой: {src_lbl}")
        return
    
    # Собираем пары (image, label)
    pairs = []
    for img_path in src_img.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl_path = src_lbl / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"⚠️ Нет разметки для {img_path.name}, пропускаем")
            continue
        pairs.append((img_path, lbl_path))
    
    if not pairs:
        print("❌ Нет валидных пар фото+разметка")
        return
    
    print(f"🔍 Найдено пар: {len(pairs)}")
    
    # Перемешиваем и делим
    random.shuffle(pairs)
    train_count = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:]
    
    print(f"📊 Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    
    # Создаём структуру
    for split in ["train", "val"]:
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    def process(pairs_list, split):
        img_dst = dst / "images" / split
        lbl_dst = dst / "labels" / split
        for i, (img_src, lbl_src) in enumerate(pairs_list, 1):
            img_dst_path = img_dst / img_src.name
            lbl_dst_path = lbl_dst / lbl_src.name
            
            if MODE == "copy":
                shutil.copy2(img_src, img_dst_path)
                shutil.copy2(lbl_src, lbl_dst_path)
            else:
                shutil.move(str(img_src), str(img_dst_path))
                shutil.move(str(lbl_src), str(lbl_dst_path))
            
            if i % 50 == 0 or i == len(pairs_list):
                print(f"   {split}: {i}/{len(pairs_list)}")
    
    print(f"📁 Копируем в train...")
    process(train_pairs, "train")
    
    print(f"📁 Копируем в val...")
    process(val_pairs, "val")
    
    print(f"\n✅ Готово! Структура:")
    print(f"   {dst}/images/train  — {len(train_pairs)} фото")
    print(f"   {dst}/images/val    — {len(val_pairs)} фото")
    print(f"   {dst}/labels/train  — {len(train_pairs)} txt")
    print(f"   {dst}/labels/val    — {len(val_pairs)} txt")

if __name__ == "__main__":
    main()