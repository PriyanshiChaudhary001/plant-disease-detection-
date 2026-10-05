"""Feature extraction for classic ML: color histogram + HOG, and leaf-damage estimate."""
import cv2
import numpy as np
from skimage.feature import hog

IMG_SIZE = 128


def extract_features(img_bgr):
    """Return one feature vector (HSV color histogram + HOG texture) for an image."""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    # 1) Color features: 8x8x8 HSV histogram (diseases change leaf color)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # 2) Texture/shape features: HOG on grayscale (spots, edges, patterns)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2))
    return np.hstack([hist, hog_feat])


def estimate_affected_area(img_bgr):
    """Approximate % of the leaf that is NOT healthy green (simple HSV segmentation).
    This is an estimate, not a medical-grade measurement."""
    img = cv2.resize(img_bgr, (256, 256))
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    leaf = (s > 40) & (v > 40)                      # leaf pixels (not grey background)
    green = leaf & (h >= 35) & (h <= 85)            # healthy green pixels
    leaf_px = int(leaf.sum())
    if leaf_px == 0:
        return 0.0
    return round(100.0 * (leaf_px - int(green.sum())) / leaf_px, 2)
