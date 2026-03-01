#!/usr/bin/env python3
"""
Convert labelme polygon annotations to YOLO bounding box format.
Performs 70/20/10 train/val/test split.
"""

import json
import shutil
import random
from pathlib import Path

DATASET_DIR = Path("car_coupling_train")
OUTPUT_DIR = Path("yolo_dataset")
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.2
TEST_RATIO  = 0.1
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    cx = (xmin + xmax) / 2 / img_w
    cy = (ymin + ymax) / 2 / img_h
    w  = (xmax - xmin) / img_w
    h  = (ymax - ymin) / img_h
    return cx, cy, w, h


def main():
    json_files = sorted(DATASET_DIR.glob("*.json"))

    pairs = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        img_path = None
        for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG"]:
            candidate = DATASET_DIR / (jf.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"WARNING: No image found for {jf.name}, skipping.")
            continue

        img_w = data["imageWidth"]
        img_h = data["imageHeight"]

        yolo_lines = []
        for shape in data.get("shapes", []):
            if shape.get("label") == "joint":
                xmin, ymin, xmax, ymax = polygon_to_bbox(shape["points"])
                cx, cy, w, h = bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if yolo_lines:
            pairs.append((img_path, yolo_lines))
        else:
            print(f"WARNING: No 'joint' labels in {jf.name}, skipping.")

    print(f"Found {len(pairs)} valid annotated images.")

    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:n_train + n_val]
    test_pairs  = pairs[n_train + n_val:]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    def write_split(pairs, split):
        for img_path, yolo_lines in pairs:
            shutil.copy2(img_path, OUTPUT_DIR / "images" / split / img_path.name)
            label_path = OUTPUT_DIR / "labels" / split / (img_path.stem + ".txt")
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

    write_split(train_pairs, "train")
    write_split(val_pairs,   "val")
    write_split(test_pairs,  "test")

    yaml_content = f"""path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: joint
"""
    with open(OUTPUT_DIR / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Dataset prepared in '{OUTPUT_DIR}/'")
    print(f"YAML config: {OUTPUT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
