"""
Predict a plant-image class using the basic NumPy/OpenCV model.
Educational demo only; results are not a reliable plant-health diagnosis.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def extract_features(image_path):
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Predict a plant class with the basic model.")
    parser.add_argument("--image", required=True, help="Path to a leaf image.")
    parser.add_argument("--model", default="plant_model_basic.npz", help="Saved model file.")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path.resolve()} (run train_basic.py first)")
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path.resolve()}")

    try:
        model = np.load(model_path, allow_pickle=False)
        centroids = model["centroids"].astype(np.float32)
        classes = model["classes"].astype(str)
        feature = extract_features(image_path)
    except (ValueError, KeyError, OSError, cv2.error) as exc:
        raise SystemExit(f"Could not load model or image: {exc}")

    distances = np.linalg.norm(centroids - feature[None, :], axis=1)
    order = np.argsort(distances)
    best = order[0]
    print(f"Predicted class: {classes[best]}")
    print("Closest classes (distance is not a confidence score):")
    for idx in order[:min(3, len(order))]:
        print(f"  {classes[idx]}: distance {distances[idx]:.4f}")
    print("Educational result only. Do not use this output as a diagnosis.")


if __name__ == "__main__":
    main()
