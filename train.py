#!/usr/bin/env python3

import shutil
import sys
from pathlib import Path

import numpy as np

from ultralytics import YOLO
from ultralytics.utils import SETTINGS


VAL_IMAGES_DIR = Path("yolo_dataset/images/val")
VAL_LABELS_DIR = Path("yolo_dataset/labels/val")
BEST_MODEL_DIR = Path("runs/train/coupling_yolov8n/weights")


def load_gt_x(image_path: Path) -> int | None:
    """Load ground truth x-center from YOLO label file."""
    label_path = VAL_LABELS_DIR / (image_path.stem + ".txt")
    if not label_path.exists():
        return None
    with open(label_path) as f:
        line = f.readline().strip()
    if not line:
        return None
    # YOLO format: class cx cy w h (normalized)
    parts = line.split()
    cx_norm = float(parts[1])
    # We need image width to denormalize
    # Read from image
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img_w = img.shape[1]
    return int(cx_norm * img_w)


def compute_val_mae(model) -> float:

    SETTINGS["runs_dir"] = "tmp/cc_eval"
    val_images = sorted(VAL_IMAGES_DIR.glob("*.*"))
    errors = []

    for img_path in val_images:
        gt_x = load_gt_x(img_path)
        if gt_x is None:
            continue

        results = model(str(img_path), conf=0.1, save=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            img_w = results[0].orig_shape[1]
            errors.append(img_w)
            continue

        best_idx = boxes.conf.cpu().numpy().argmax()
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        pred_x = int((xyxy[0] + xyxy[2]) / 2)
        errors.append(abs(pred_x - gt_x))

    return float(np.mean(errors)) if errors else float("inf")


def main():

    model = YOLO("yolov8n.pt")

    best_mae = float("inf")
    best_epoch = -1
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_pt_path = BEST_MODEL_DIR / "best.pt"

    def on_epoch_end(trainer):
        nonlocal best_mae, best_epoch

        epoch = trainer.epoch + 1

        last_pt = Path(trainer.save_dir) / "weights" / "last.pt"
        if not last_pt.exists():
            return

        eval_model = YOLO(str(last_pt))
        mae = compute_val_mae(eval_model)

        print(
            f"  [x-MAE] epoch {epoch}: MAE={mae:.1f}px  (best so far: {best_mae:.1f}px @ epoch {best_epoch})"
        )

        print(last_pt)

        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            shutil.copy2(last_pt, best_pt_path)
            print(f"  [x-MAE] *** New best model saved (MAE={mae:.1f}px) ***")

    model.add_callback("on_train_epoch_end", on_epoch_end)

    model.train(
        data="yolo_dataset/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        patience=20,
        augment=True,
        degrees=5,
        translate=0.1,
        scale=0.3,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.5,
        project="runs/train",
        name="coupling_yolov8n",
        exist_ok=True,
        device="",
        workers=4,
        verbose=True,
        save_period=-1,
    )

    print(f"\nTraining complete!")
    print(f"Best model: epoch {best_epoch}, MAE={best_mae:.1f}px")
    print(f"Saved to: {best_pt_path}")


if __name__ == "__main__":
    main()
