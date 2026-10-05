"""Train a Random Forest on color-histogram + HOG features of leaf images.

Usage:
    python train.py --data_dir data --filter tomato --max_per_class 300
"""
import argparse
import os
import random

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from features import extract_features


def load_data(data_dir, name_filter, max_per_class):
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d))
                     and name_filter.lower() in d.lower())
    if not classes:
        raise SystemExit(f"No class folders found in '{data_dir}' matching '{name_filter}'")
    X, y = [], []
    for idx, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.Random(42).shuffle(files)
        count = 0
        for f in files[:max_per_class]:
            img = cv2.imread(os.path.join(folder, f))
            if img is None:
                continue
            X.append(extract_features(img))
            y.append(idx)
            count += 1
        print(f"[{idx}] {cls}: {count} images loaded")
    return np.array(X), np.array(y), classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--filter", default="", help="only use class folders containing this text, e.g. tomato")
    ap.add_argument("--max_per_class", type=int, default=300)
    args = ap.parse_args()

    X, y, classes = load_data(args.data_dir, args.filter, args.max_per_class)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("\nTraining Random Forest ...")
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    report = classification_report(y_te, pred, target_names=classes)
    print(f"\nTest accuracy: {acc * 100:.2f}%\n")
    print(report)

    joblib.dump({"model": clf, "classes": classes}, "plant_model.pkl")
    with open("results.txt", "w") as f:
        f.write(f"Test accuracy: {acc * 100:.2f}%\n\n{report}")

    cm = confusion_matrix(y_te, pred)
    short = [c.replace("___", " ").replace("_", " ") for c in classes]
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=short, yticklabels=short)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix - Random Forest")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved: plant_model.pkl, results.txt, confusion_matrix.png")


if __name__ == "__main__":
    main()
