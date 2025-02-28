
import numpy as np

mtx = np.load('mtx.npy')
dist = np.load('dist.npy')
newcameramtx = np.load('newcameramtx.npy')
roi = np.load('roi.npy')

print("mtx:", mtx)  # 内参数矩阵
print("dist:", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("newcameramtx:", newcameramtx)
print("roi:", roi)
