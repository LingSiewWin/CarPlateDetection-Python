# ocr_engine.py

import easyocr
import numpy as np

# Create a global EasyOCR reader instance (English, GPU off)
reader = easyocr.Reader(['en'], gpu=False)

def recognize_character(char_img):
    """
    Use EasyOCR to recognize a single character image.

    Args:
        char_img (np.array): Single character binary or grayscale image

    Returns:
        str: Recognized character
    """
    if len(char_img.shape) == 2:
        img_rgb = np.stack([char_img]*3, axis=-1)
    else:
        img_rgb = char_img
    result = reader.readtext(img_rgb, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    if result:
        return result[0].strip()
    return ''


def postprocess_ocr_result(text):
    """
    Post-process the OCR result, including context correction and format normalization.

    Args:
        text (str): Raw OCR output string

    Returns:
        str: Corrected result
    """
    corrections = {
        '0': 'O', '1': 'I', '8': 'B', '5': 'S',
        '6': 'G', '2': 'Z', 'Q': '0'
    }

    corrected = ''.join([corrections[c] if c in corrections else c for c in text.upper()])
    return ''.join([c for c in corrected if c.isalnum()])