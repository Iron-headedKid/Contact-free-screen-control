import cv2
import numpy as np
import glob


# -----------畸变校正函数--------------
def recify():
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    images = glob.glob("D:\pict1/*.jpg")
    i = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
        # print(corners)

        if ret:

            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            # print(corners2)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)

            cv2.drawChessboardCorners(img, (7, 7), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
            i += 1
            cv2.imwrite('qipan' + str(i) + '.jpg', img)
            cv2.waitKey(1500)

    print(len(img_points))
    cv2.destroyAllWindows()

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    # print("ret:", ret)
    # print("mtx:", mtx)  # 内参数矩阵
    # print("dist:", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    # print("rvecs:", rvecs)  # 旋转向量  # 外参数
    # print("tvecs:", tvecs)  # 平移向量  # 外参数

    # print("---------使用getOptimalNewCameraMatrix函数----------------")
    img = cv2.imread(images[2])
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))  # 显示更大范围的图片（正常重映射之后会删掉一部分图像）
    # print("newcameramtx:",newcameramtx)
    # print("roi:",roi)
    # print("------------------使用undistort函数-------------------")
    # dst = cv2.undistort(target, mtx, dist, None, newcameramtx)
    # x, y, w, h = roi
    # dst1 = dst[y:y + h, x:x + w]
    return mtx, dist,newcameramtx,roi

# -------------鼠标点击响应-------------
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,0), thickness = 1)
        pointss.append((x,y))
        cv2.imshow("image", img)

def find_opints(img,opints):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    while (1):
        cv2.imshow("image", img)
        if cv2.waitKey(0) & 0xFF == 27: #esc退出
            break
    cv2.destroyAllWindows()
    print(pointss)#测试用
    # 选取四个点，分别是左上、右上、左下、右下
    points, img = draw_line(img, pointss[0], pointss[1], pointss[2], pointss[3])


#----------透射变化模块-------------------------------
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
    print(M)
    # 透视逆矩阵
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    print(M_inverse)
    return M, M_inverse

# 透视变换
def img_perspect_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size)

def draw_line(img,p1,p2,p3,p4):
    points = [list(p1), list(p2), list(p3), list(p4)]
    # 画线
    img = cv2.line(img, p1, p2, (0, 0, 255), 3)
    img = cv2.line(img, p2, p4, (0, 0, 255), 3)
    img = cv2.line(img, p4, p3, (0, 0, 255), 3)
    img = cv2.line(img, p3, p1, (0, 0, 255), 3)
    return points,img



if __name__ == '__main__':
    # ----------首次计算畸变校正参数---------------------
    #cv2.imshow('test01', target)
    roi=()
    mtx, dist,newcameramtx,roi = recify()#校正函数
    print("mtx:", mtx)  # 内参数矩阵
    np.save('mtx.npy', mtx)
    print("dist:", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    np.save('dist.npy', dist)
    print("newcameramtx:", newcameramtx)
    np.save('newcameramtx.npy', newcameramtx)
    print("roi:", roi)
    np.save('roi.npy', roi)
    pointss = []#存储选取的四个屏幕点


#------------循环开始的地方，自己加循环---------------------------
    target = cv2.imread('D:\pict\screen.jpg') #校正目标图
    dst = cv2.undistort(target, mtx, dist, None, newcameramtx)#校正函数
    x, y, w, h = roi
    img = dst[y:y + h, x:x + w]
    cv2.imwrite('dest.jpg', img)
    #cv2.imshow("perspective_img",dst1)
    #cv2.waitKey(0)
    if len(pointss)==0:#只有第一次要选点
        find_opints(img,pointss)#选点函数
    #----------------做透射变化---------------
    cv2.imshow('test01',img)
    cv2.waitKey(0)
    cv2.imwrite('test01.png',img)
    M, M_inverse = cal_perspective_params(img, pointss)#计算变化矩阵
    trasform_img = img_perspect_transform(img, M)#透射变化
    cv2.imshow('test02.png',trasform_img)
    cv2.waitKey(0)
    cv2.imwrite('test02.png',trasform_img)
