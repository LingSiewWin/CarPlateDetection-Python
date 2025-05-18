# ocr_engine.py

import easyocr
import numpy as np

# Create a global EasyOCR reader instance (English, GPU off)
reader = easyocr.Reader(['en'], gpu=False)

def recognize_character(char_img):
    """
    使用 EasyOCR 识别一个字符图像。

    参数:
        char_img (np.array): 单个字符的二值或灰度图像

    返回:
        str: 识别出的字符
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
    对OCR结果进行后处理,包括上下文纠错和格式标准化。

    参数:
        text (str): OCR原始输出字符串

    返回:
        str: 纠错后的结果
    """
    corrections = {
        '0': 'O', '1': 'I', '8': 'B', '5': 'S',
        '6': 'G', '2': 'Z', 'Q': '0'
    }

    corrected = ''.join([corrections[c] if c in corrections else c for c in text.upper()])
    return ''.join([c for c in corrected if c.isalnum()])