import cv2
import numpy as np
import find_finger as ff
import operation_judge as oj
import pic_processing as pp


# Ofad == operating from a distance
class Ofad(object):
    def __init__(self):
        self.capture1 = cv2.VideoCapture(0)     # 选择摄像头
        self.capture1.set(3, 1280)  # 设置宽度
        self.capture1.set(4, 720)  # 设置高度
        self.x0 = 0                 # 判定坐标
        self.y0 = 0                 # 判定坐标
        self.points = []            # 操作面坐标

    def run(self):
        self.points = np.load('points.npy')
        # 稳定运行阶段
        while True:
            ret, frame = self.capture1.read()
            frame = pp.processing(frame, self.points)       # 透射变换
            crop, x, y = ff.process(frame)                  # 找到并定位手指（或某物体）
            self.x0, self.y0 = oj.judge(x, y, self.x0, self.y0)     # 操作判定
            cv2.imshow("camera", crop)
            key = cv2.waitKey(1)  # 等待键盘输入一毫秒
            if key == 27:  # 按esc键退出
                break

        cv2.destroyAllWindows()
        self.capture1.release()


if __name__ == "__main__":
    Ofad().run()
