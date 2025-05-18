# color_segmentation.py

import cv2
import numpy as np

def find_malaysia_plates(img):
    """
    根据马来西亚车牌的颜色特征提取车牌候选区域。

    支持以下组合：
    - 黄底黑字(Commercial)
    - 红底白字(Diplomatic)
    - 绿底白字(Military)
    - 蓝底白字(Government)

    参数:
        img (np.array): 输入的BGR图像

    返回:
        np.array: 提取后的车牌候选区域图像
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义各类型车牌的颜色范围
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

    # 创建掩膜
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # 合并所有掩膜
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask_yellow, mask_red),
                                   cv2.bitwise_or(mask_green, mask_blue))

    # 应用掩膜提取车牌区域
    result = cv2.bitwise_and(img, img, mask=combined_mask)

    return result