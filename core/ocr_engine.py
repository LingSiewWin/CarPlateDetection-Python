# ocr_engine.py

import pytesseract
from PIL import Image

def recognize_character(char_img):
    """
    使用 Tesseract OCR 识别一个字符图像。

    参数:
        char_img (np.array): 单个字符的二值图像（黑底白字）

    返回:
        str: 识别出的字符
    """
    pil_img = Image.fromarray(char_img)
    config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    try:
        text = pytesseract.image_to_string(pil_img, config=config).strip()
    except:
        text = ''
    return text


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