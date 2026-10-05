"""Analyze one leaf image: disease, confidence %, affected area %, precaution, cure.

Usage:
    python predict.py --image path/to/leaf.jpg
"""
import argparse
import textwrap

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from disease_info import get_info
from features import estimate_affected_area, extract_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="plant_model.pkl")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    clf, classes = bundle["model"], bundle["classes"]

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit("Could not read the image. Check the path.")

    probs = clf.predict_proba([extract_features(img)])[0]
    order = np.argsort(probs)[::-1]
    top = order[0]
    disease = classes[top].replace("___", " - ").replace("_", " ")
    confidence = probs[top] * 100
    area = estimate_affected_area(img)
    info = get_info(classes[top])

    lines = [
        f"Disease        : {disease}",
        f"Confidence     : {confidence:.2f}%",
        f"Affected leaf  : ~{area}% (estimated)",
        f"Severity       : {info['severity']}",
        "",
        "Top 3 predictions:",
    ]
    for i in order[:3]:
        lines.append(f"  - {classes[i].replace('___', ' - ').replace('_', ' ')}: {probs[i] * 100:.2f}%")
    lines += ["", "Precaution:", textwrap.fill(info["precaution"], 55), "",
              "Cure:", textwrap.fill(info["cure"], 55)]
    text = "\n".join(lines)
    print("\n" + text + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].axis("off")
    ax[1].axis("off"); ax[1].text(0, 1, text, va="top", family="monospace", fontsize=10)
    plt.tight_layout(); plt.savefig("prediction_output.png", dpi=150)
    print("Saved: prediction_output.png (use this as your README screenshot)")


if __name__ == "__main__":
    main()
