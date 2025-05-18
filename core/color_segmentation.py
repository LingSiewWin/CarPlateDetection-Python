# color_segmentation.py

import cv2
import numpy as np

def find_malaysia_plates(img):
    """
    Extract candidate plate regions based on the color features of Malaysian license plates.

    Supported combinations:
    - Yellow background with black text (Commercial)
    - Red background with white text (Diplomatic)
    - Green background with white text (Military)
    - Blue background with white text (Government)

    Args:
        img (np.array): Input BGR image

    Returns:
        np.array: Extracted candidate plate region image
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges for each plate type
    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])

    green_lower = np.array([40, 40, 40])
    green_upper = np.array([90, 255, 255])

    blue_lower = np.array([100, 150, 50])
    blue_upper = np.array([140, 255, 255])

    # Create masks
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # Combine all masks
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask_yellow, mask_red),
                                   cv2.bitwise_or(mask_green, mask_blue))

    # Apply mask to extract plate regions
    result = cv2.bitwise_and(img, img, mask=combined_mask)

    return result