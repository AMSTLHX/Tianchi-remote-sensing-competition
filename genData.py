import os
import cv2
from PIL import Image
import numpy as np
from skimage import io
import matplotlib.image as mpimg # mpimg 用于读取图片
Image.MAX_IMAGE_PIXELS = 100000000000

img=io.imread('./data/jingwei_round1_train_20190619/image_1.png')
print(img.shape)
img = Image.open('./data/jingwei_round1_train_20190619/image_1.png')   # 注意修改img路径
img = np.asarray(img)
print(img.shape)

anno_map = Image.open('./data/jingwei_round1_train_20190619/image_1_label.png')   # 注意修改label路径
anno_map = np.asarray(anno_map)
print(anno_map.shape)

cimg = cv2.resize(img, None, fx= 0.1, fy=0.1)
cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2BGR)
cv2.imwrite('./data/vis/train_image_1.png', cimg, [int(cv2.IMWRITE_JPEG_QUALITY),100])   # 注意修改 可视化img 的路径

B = anno_map.copy()   # 蓝色通道
B[B == 1] = 255
B[B == 2] = 0
B[B == 3] = 0
B[B == 0] = 0

G = anno_map.copy()   # 绿色通道
G[G == 1] = 0
G[G == 2] = 255
G[G == 3] = 0
G[G == 0] = 0

R = anno_map.copy()   # 红色通道
R[R == 1] = 0
R[R == 2] = 0
R[R == 3] = 255
R[R == 0] = 0

anno_vis = np.dstack((B,G,R))
anno_vis = cv2.resize(anno_vis, None, fx= 0.1, fy=0.1)
cv2.imwrite('./data/vis/train_image_1_label.png', anno_vis)   # 注意修改 可视化label 的路径

unit_size = 256   # 窗口大小

length, width = img.shape[0], img.shape[1]
x1, x2, y1, y2 = 0, unit_size ,0 ,unit_size
Img = [] # 保存小图的数组
Label = []  # 保存label的数组
while(x1 < length):
    #判断横向是否越界
    if x2 > length:
        x2 , x1 = length , length - unit_size

    while(y1 < width):
        if y2 > width:
            y2 , y1  = width , width - unit_size
        im = img[x1:x2, y1:y2, :]
        if 255 in im[:,:,-1]:    # 判断裁剪出来的小图中是否存在有像素点
            Img.append(im[:,:,0:3])   # 添加小图
            Label.append(anno_map[x1:x2, y1:y2])   # 添加label

        if y2 == width: break

        y1 += unit_size
        y2 += unit_size

    if x2 == length: break

    y1,y2 = 0 , unit_size
    x1 += unit_size
    x2 += unit_size

Img = np.array(Img)
Label = np.array(Label)

print(Img.shape)
print(Label.shape)
np.save('./data/jingwei_round1_train_20190619/image_1.npy', Img)    # 注意修改 npy-Img 的路径
np.save('./data/jingwei_round1_train_20190619/image_1_label.npy', Label)   # 注意修改 npy-label 的路径