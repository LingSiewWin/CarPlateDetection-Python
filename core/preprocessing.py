# preprocessing.py

import cv2
import numpy as np

def enhance_characters(img):
    """
    对输入的车牌图像进行增强预处理。
    
    参数:
        img (np.array): 输入图像
        
    返回:
        np.array: 增强后的图像
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # CLAHE 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 高斯模糊降噪
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 自适应阈值二值化
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 形态学闭运算连接字符
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned