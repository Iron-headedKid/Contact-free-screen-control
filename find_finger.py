import cv2
import numpy as np


def process(frame):
    crop = frame
#    print('高度：', crop.shape[0])
#    print('宽度：', crop.shape[1])
    r, thresh = cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 157, 255, cv2.THRESH_BINARY)  # 设置阈值

    # 找出轮廓；返回值：轮廓、层次结构；参数：输入图像、层次结构类型、轮廓近似方法
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = 0, 0, 0, 0
    for c in contours:
        # find bounding box coordinates
        x0, y0, w0, h0 = cv2.boundingRect(c)
        if w0 > 50:                 # 避免不必要的轮廓
            cv2.rectangle(crop, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)  # 画矩形轮廓
            x, y, w, h = x0, y0, w0, h0

    center_x = int(x + w / 2)
    center_y = int(y + h / 2)
#    print('中心点坐标为', center_x, center_y)
    cv2.circle(crop, (center_x, center_y), 2, (0, 0, 255), 2)  # 绘制中心点
    return crop, center_x, center_y
