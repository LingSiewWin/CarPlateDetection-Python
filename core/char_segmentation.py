# char_segmentation.py

import cv2
import numpy as np

def segment_characters(binary_img):
    """
    使用投影法对预处理后的图像进行字符分割。

    参数:
        binary_img (np.array): 增强后的二值图像

    返回:
        list: 分割出的字符图像列表
    """
    contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 8 < w < 100 and 15 < h < 150 and w * h > 300:
            char_candidate = binary_img[y:y+h, x:x+w]
            char_candidates.append((x, char_candidate))

    # 按照从左到右排序
    char_candidates.sort(key=lambda x: x[0])
    characters = [ch[1] for ch in char_candidates]

    return characters