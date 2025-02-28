import cv2
import numpy as np
import glob

tpPointsChoose = []
drawing = False
tempFlag = False


# -----------畸变校正函数--------------
def recify(tar, mtx, dist, newcameramtx, roi):
    dst = cv2.undistort(tar, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst1 = dst[y:y + h, x:x + w]
    return dst1


# ----------透射变化模块-------------------------------
# 计算透视变换参数矩阵
def cal_perspective_params(img, points):
    # 设置偏移点。如果设置为(0,0),表示透视结果只显示变换的部分（也就是画框的部分）
    offset_x = 100
    offset_y = 100
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(points)
    # 透视变换的四个点
    dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x, offset_y],
                      [offset_x, img_size[1] - offset_y], [img_size[0] - offset_x, img_size[1] - offset_y]])
    # 透视矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    # 透视逆矩阵
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    return M, M_inverse


# 透视变换
def img_perspect_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size)


def processing(frame, points):
    # ----------------做透射变化---------------
    M, M_inverse = cal_perspective_params(frame, points)  # 计算变化矩阵
    trasform_img = img_perspect_transform(frame, M)  # 透射变化
    return trasform_img

