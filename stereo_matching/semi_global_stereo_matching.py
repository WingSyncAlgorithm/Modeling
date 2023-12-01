import cv2
import numpy as np

rect_left_image = cv2.imread("im1_left.png")
rect_right_image = cv2.imread("im1_right.png")

# SGBM演算法參數設置
WINDOW_SIZE = 9
MIN_DISPARITY = 0
NUM_DISPARITIES = 64
BLOCK_SIZE = 9
P1 = 8 * 3 * WINDOW_SIZE**2
P2 = 32 * 3 * WINDOW_SIZE**2
DISP12_MAX_DIFF = 1
PRE_FILTER_CAP = 63
UNIQUENESS_RATIO = 10
SPECKLE_WINDOW_SIZE = 5
SPECKLE_RANGE = 32

sgbm = cv2.StereoSGBM_create(minDisparity=MIN_DISPARITY,
                             numDisparities=NUM_DISPARITIES,
                             blockSize=BLOCK_SIZE,
                             P1=P1,
                             P2=P2,
                             disp12MaxDiff=DISP12_MAX_DIFF,
                             preFilterCap=PRE_FILTER_CAP,
                             uniquenessRatio=UNIQUENESS_RATIO,
                             speckleWindowSize=SPECKLE_WINDOW_SIZE,
                             speckleRange=SPECKLE_RANGE,
                             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

# 計算視差圖
disparity = sgbm.compute(rect_left_image, rect_right_image)
disparity_nor = cv2.normalize(disparity,
                              disparity,
                              alpha=0,
                              beta=255,
                              norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8U)

# WLS濾波參數設置
LAMBDA_VAL = 8000
SIGMA_VAL = 1.5

# 運行WLS濾波
wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
wls_filter.setLambda(LAMBDA_VAL)
wls_filter.setSigmaColor(SIGMA_VAL)
filtered_disp = wls_filter.filter(disparity, rect_left_image, None,
                                  rect_right_image)
filtered_disp_nor = cv2.normalize(filtered_disp,
                                  filtered_disp,
                                  alpha=0,
                                  beta=255,
                                  norm_type=cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8U)

cv2.imwrite(
    'disparity.jpg',
    cv2.resize(disparity_nor,
               (disparity_nor.shape[1] // 2, disparity_nor.shape[0] // 2)))
cv2.imwrite(
    'filtered_disparity.jpg',
    cv2.resize(
        filtered_disp_nor,
        (filtered_disp_nor.shape[1] // 2, filtered_disp_nor.shape[0] // 2)))