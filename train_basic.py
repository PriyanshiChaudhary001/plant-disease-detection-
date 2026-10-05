"""
Basic plant-image classifier using only NumPy and OpenCV.
Dataset layout: data/<class_name>/<image files>
Example: data/Tomato___Early_blight/image1.jpg
This is an educational demo, not a reliable plant-health diagnosis.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_features(image_path):
    # Read paths safely, including folders with spaces or non-ASCII characters.
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train a simple NumPy/OpenCV plant classifier.")
    parser.add_argument("--data_dir", default="data", help="Folder containing one subfolder per class.")
    parser.add_argument("--filter", default=None, help="Optional class-name substring, e.g. tomato.")
    parser.add_argument("--max_per_class", type=int, default=None, help="Optional image limit per class.")
    parser.add_argument("--output", default="plant_model_basic.npz", help="Output model file.")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.is_dir():
        raise SystemExit(f"Dataset folder not found: {root.resolve()}")

    features, labels = [], []
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.filter:
        class_dirs = [p for p in class_dirs if args.filter.lower() in p.name.lower()]
    if not class_dirs:
        raise SystemExit("No class folders found. Expected data/<class_name>/<images>.")

    for class_dir in class_dirs:
        count = 0
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if args.max_per_class is not None and count >= args.max_per_class:
                break
            try:
                features.append(extract_features(path))
                labels.append(class_dir.name)
                count += 1
            except (ValueError, cv2.error) as exc:
                print(f"Skipping {path}: {exc}")
        print(f"{class_dir.name}: {count} images")

    if not features:
        raise SystemExit("No readable images found in the selected class folders.")

    x = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=str)
    classes = np.unique(y)
    centroids = np.vstack([x[y == name].mean(axis=0) for name in classes]).astype(np.float32)

    np.savez_compressed(args.output, centroids=centroids, classes=classes)
    print(f"Saved basic model to: {Path(args.output).resolve()}")
    print(f"Classes: {len(classes)} | Images used: {len(y)}")
    print("This simple color-histogram model is for learning, not diagnosis.")


if __name__ == "__main__":
    main()
