import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

MAX_PARALLAX = 25 #最大視差
WIN_SIZE = 5 #滑動窗口大小

os.chdir('C:/Users/user/Desktop/vs code/局部立體匹配')
left_image = np.asanyarray(Image.open(r'left.jpg')) #左相機圖
right_image = np.asanyarray(Image.open(r'right.jpg')) #右相機圖

left_image = cv2.cvtColor(left_image,cv2.COLOR_BGR2GRAY) #轉為灰度圖
right_image = cv2.cvtColor(right_image,cv2.COLOR_BGR2GRAY)
left_image = cv2.medianBlur(left_image, 5) #加入中值濾波
right_image = cv2.medianBlur(right_image, 5)
left_image = np.asanyarray(left_image,dtype=np.double) #轉為double型態
right_image = np.asanyarray(right_image,dtype=np.double) 

image_size = np.shape(left_image)[0:2] #定義和圖寬度、高度相等的數組
cv2.imshow('左圖灰度圖', left_image)
cv2.imwrite('left灰.jpg', left_image)
cv2.imshow('右圖灰度圖', right_image)
cv2.imwrite('right灰.jpg', right_image)

image_diff = np.zeros((image_size[0],image_size[1],MAX_PARALLAX))
e = np.zeros(image_size)
for i in range(0,MAX_PARALLAX):
    e = np.abs(right_image[:,0:(image_size[1]-i)]-left_image[:,i:image_size[1]])
    e2 = np.zeros(image_size) 
    for x in range(0,image_size[0]):
        for y in range(0,image_size[1]):
            e2[x,y] = np.sum(e[(x-WIN_SIZE):(x+WIN_SIZE), (y-WIN_SIZE):(y+WIN_SIZE)])
    image_diff[:,:,i] = e2
dispmap = np.zeros(image_size) #最小視差圖
for x in range(0,image_size[0]):
    for y in range (0,image_size[1]):
        val = np.sort(image_diff[x,y,:])
        if np.abs(val[0]-val[1]) > 10:
            val_id = np.argsort(image_diff[x,y,:])
            dispmap[x,y] = val_id[0]/MAX_PARALLAX*255 #恢復彩色

# 加入雙邊濾波
denoised_dispmap = cv2.bilateralFilter(dispmap.astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)

cv2.imshow('視差圖', denoised_dispmap)
cv2.imwrite('diff.jpg', denoised_dispmap)
plt.show
