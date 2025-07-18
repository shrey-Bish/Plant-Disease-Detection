import mahotas
import cv2
import numpy as np

def extract_features(image):
    # Color histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Hu moments
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Haralick
    haralick = mahotas.features.haralick(gray).mean(axis=0)

    return np.hstack([hist, hu_moments, haralick])
