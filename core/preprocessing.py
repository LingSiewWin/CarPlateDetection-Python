# preprocessing.py

import cv2
import numpy as np

def enhance_characters(img):
    """
    Enhance the input license plate image.
    
    Args:
        img (np.array): Input image
        
    Returns:
        np.array: Enhanced image
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Gaussian blur for denoising
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Adaptive thresholding for binarization
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological closing to connect characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned