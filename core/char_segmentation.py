# char_segmentation.py

import cv2
import numpy as np
import re

def segment_characters(binary_img):
    """
    Use projection method to segment characters from the preprocessed binary image.

    Args:
        binary_img (np.array): Enhanced binary image

    Returns:
        list: List of segmented character images
    """
    contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 8 < w < 100 and 15 < h < 150 and w * h > 300:
            char_candidate = binary_img[y:y+h, x:x+w]
            char_candidates.append((x, char_candidate))

    # Sort from left to right
    char_candidates.sort(key=lambda x: x[0])
    characters = [ch[1] for ch in char_candidates]

    return characters

def preprocess_ocr_text(text):
    text = text.upper().replace(' ', '').replace('-', '')

    # Only correct the first 1-3 chars (prefix)
    prefix_raw = text[:3]
    number_raw = text[3:]

    # Corrections for prefix only
    prefix_corrections = {
        '0': 'O', '1': 'I', '8': 'B', '5': 'S', '6': 'G', '2': 'Z', 'Q': 'O',
        '13': 'B'
    }
    for wrong, right in prefix_corrections.items():
        prefix_raw = prefix_raw.replace(wrong, right)
    prefix = ''.join(c for c in prefix_raw if c.isalpha())

    # For the number part, keep only digits, no correction
    numbers = ''.join(c for c in number_raw if c.isdigit())

    return prefix + numbers

def is_valid_plate_format(plate):
    return re.fullmatch(r'[A-Z]{1,3}[0-9]{1,4}', plate) is not None