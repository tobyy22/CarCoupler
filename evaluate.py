#!/usr/bin/env python3
"""
Evaluate the trained model on validation/test images and draw predictions.

Usage:
    python evaluate.py                          # uses val split
    python evaluate.py --split test             # final evaluation on test split
    python evaluate.py --split train
    python evaluate.py img1.jpg img2.jpg        # specific images
"""

import argparse
import sys
from pathlib import Path
import json

import cv2
import numpy as np

from ultralytics import YOLO

DATASET_DIR   = Path("car_coupling_train")
DEFAULT_MODEL = "runs/train/coupling_yolov8n/weights/best.pt"
DEFAULT_SAVE_DIR = Path("eval_results")


def load_ground_truth(image_path: Path):
    """Load GT from original labelme JSON in car_coupling_train/."""
    json_path = DATASET_DIR / (image_path.stem + ".json")
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    for shape in data.get("shapes", []):
        if shape.get("label") == "joint":
            points = shape["points"]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_center = int((min(xs) + max(xs)) / 2)
            y_center = int((min(ys) + max(ys)) / 2)
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            return x_center, y_center, bbox
    return None


def draw_result(image_path: Path, model, save_dir: Path, conf_thresh: float = 0.1):
    # Read image from original dataset dir
    img = cv2.imread(str(DATASET_DIR / image_path.name))
    if img is None:
        # Fallback to provided path
        img = cv2.imread(str(image_path))
    if img is None:
        print(f"WARNING: Could not read {image_path.name}")
        return None

    h, w = img.shape[:2]

    # Ground truth
    gt = load_ground_truth(image_path)
    if gt is not None:
        gt_x, gt_y, (gx1, gy1, gx2, gy2) = gt
        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 200, 0), 2)
        cv2.line(img, (gt_x, 0), (gt_x, h), (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"GT x={gt_x}", (gx1, gy1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    # Prediction — run on original image
    src_img = str(DATASET_DIR / image_path.name)
    results = model(src_img, conf=conf_thresh, save=False, verbose=False)
    pred_x = None
    pred_error = None

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        best_idx = boxes.conf.cpu().numpy().argmax()
        xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        conf = float(boxes.conf[best_idx].cpu())

        px1, py1, px2, py2 = xyxy
        pred_x = (px1 + px2) // 2

        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 220), 2)
        cv2.line(img, (pred_x, 0), (pred_x, h), (0, 80, 220), 1, cv2.LINE_AA)
        cv2.putText(img, f"PRED x={pred_x} ({conf:.2f})", (px1, py2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

        if gt is not None:
            pred_error = abs(pred_x - gt_x)
            cv2.putText(img, f"err={pred_error}px", (px1, py2 + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)
    elif gt is not None:
        cv2.putText(img, "NO DETECTION", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 220), 3)

    cv2.putText(img, "GREEN=GT  RED=PRED", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    out_path = save_dir / f"eval_{image_path.stem}.jpg"
    cv2.imwrite(str(out_path), img)

    return pred_x, pred_error


def main():
    parser = argparse.ArgumentParser(description="Evaluate car coupling detector")
    parser.add_argument("images", nargs="*", help="Image files to evaluate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--conf", type=float, default=0.1)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}. Run train.py first.")
        sys.exit(1)

    model = YOLO(str(model_path))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.images:
        image_paths = [Path(p) for p in args.images]
    else:
        split_dir = Path("yolo_dataset/images") / args.split
        if not split_dir.exists():
            print(f"Split dir not found: {split_dir}. Run prepare_dataset.py first.")
            sys.exit(1)
        image_paths = sorted(split_dir.glob("*.*"))
        print(f"Evaluating {len(image_paths)} images from '{args.split}' split...")

    errors = []
    missing = 0

    for img_path in image_paths:
        result = draw_result(img_path, model, save_dir, conf_thresh=args.conf)
        if result is None:
            continue
        pred_x, error = result
        if pred_x is None:
            missing += 1
            print(f"  {img_path.name}: NO DETECTION")
        elif error is not None:
            errors.append(error)
            print(f"  {img_path.name}: pred_x={pred_x}  err={error}px")
        else:
            print(f"  {img_path.name}: pred_x={pred_x}  (no GT)")

    print("\n" + "=" * 50)
    print(f"Images evaluated : {len(image_paths)}")
    print(f"No detection     : {missing}")
    if errors:
        print(f"Mean abs error   : {np.mean(errors):.1f} px")
        print(f"Median abs error : {np.median(errors):.1f} px")
        print(f"Max abs error    : {np.max(errors):.1f} px")
        print(f"% within 50px    : {100*np.mean(np.array(errors) <= 50):.1f}%")
        print(f"% within 100px   : {100*np.mean(np.array(errors) <= 100):.1f}%")
    print(f"\nImages saved to  : {save_dir}/")


if __name__ == "__main__":
    main()
