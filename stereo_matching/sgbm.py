import cv2
import numpy as np
from PIL import Image
import os

# 取得桌面路徑
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

# 切換目錄至 "vscode" 資料夾中的 局部立體匹配" 資料夾
project_path = os.path.join(desktop_path, 'vs code', '局部立體匹配')
os.chdir(project_path)


left_img = np.asarray(Image.open('left1.png'))  # 左相機圖
right_img = np.asarray(Image.open('right1.png'))  # 右相機圖

sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=96,
    blockSize=9,
    P1=8 * 9 * 9,
    P2=32 * 9 * 9,
    disp12MaxDiff=1,
    uniquenessRatio=63,
    speckleWindowSize=10,
    speckleRange=100,
    mode=cv2.STEREO_SGBM_MODE_HH,
)
disparity = sgbm.compute(left_img, right_img)


cv2.imshow("disparity",disparity)
cv2.imwrite('sgbm_disparity.jpg',disparity)
cv2.waitKey(0)