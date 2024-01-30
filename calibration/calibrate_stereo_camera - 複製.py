import numpy as np
import cv2 as cv
import glob

def stereo_calibrate_from_parameters():
    """使用提供的左右相機參數進行雙目標定"""

    # 讀取左相機參數
    new_camera_matrix_left = np.array([[361.39447429, 0, 171.49965844],
                                       [0, 369.83559204, 153.02054912],
                                       [0, 0, 1]])

    dist_coeffs_left = np.array([[0.0398514, 0.48694156, 0.02214034, 0.00875936, -1.32645063]])

    # 讀取右相機參數
    new_camera_matrix_right = np.array([[353.35291965, 0, 165.83908054],
                                        [0, 301.41518211, 135.60732919],
                                        [0, 0, 1]])

    dist_coeffs_right = np.array([[0.108822493, -3.8571189, 0.00950847624, 0.00833250108, 21.7705452]])

    # 检查矩阵的形状
    assert new_camera_matrix_left.shape == (3, 3)
    assert dist_coeffs_left.shape == (1, 5)
    assert new_camera_matrix_right.shape == (3, 3)
    assert dist_coeffs_right.shape == (1, 5)

    # 建立標定用的物點座標
    CHECKER_BOARD = (6, 9)  # 棋盤格內角點(格子長、寬各自減一)
    object_points = np.zeros((CHECKER_BOARD[0] * CHECKER_BOARD[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:CHECKER_BOARD[0], 0:CHECKER_BOARD[1]].T.reshape(-1, 2)

    # 讀取影像
    images_left = sorted(glob.glob('left\\*.jpg'))
    images_right = sorted(glob.glob('right\\*.jpg'))

    # 建立儲存角點座標的列表
    img_points_left = []
    img_points_right = []

    for image_left, image_right in zip(images_left, images_right):
        img_left = cv.imread(image_left)
        img_right = cv.imread(image_right)
        gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

        # 棋盤角點檢測
        return_find_corner_left, corners_left = cv.findChessboardCorners(
            gray_left, CHECKER_BOARD, None)
        return_find_corner_right, corners_right = cv.findChessboardCorners(
            gray_right, CHECKER_BOARD, None)

        if return_find_corner_left and return_find_corner_right:
            img_points_left.append(corners_left)
            img_points_right.append(corners_right)

    # 進行雙目標定
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    flags = 0
    flags = cv.CALIB_FIX_INTRINSIC

    return_stereo, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
        [object_points] * len(images_left), img_points_left, img_points_right,
        new_camera_matrix_left, dist_coeffs_left, new_camera_matrix_right,
        dist_coeffs_right, gray_left.shape[::-1], criteria_stereo, flags)

    # 寫入相機參數到 camera_parameters.txt
    f = open("camera_parameters.txt", 'w')

    print(return_stereo, file=f)
    print("\nCamera matrix left: ", file=f)
    print(new_camera_matrix_left, file=f)
    print("\ndistCoeffs left: ", file=f)
    print(dist_coeffs_left, file=f)
    print("\nCamera matrix right: ", file=f)
    print(new_camera_matrix_right, file=f)
    print("\ndistCoeffs right: ", file=f)
    print(dist_coeffs_right, file=f)
    print("\nR: ", file=f)
    print(R, file=f)
    print("\nT: ", file=f)
    print(T, file=f)
    print("\nE: ", file=f)
    print(E, file=f)
    print("\nF: ", file=f)
    print(F, file=f)

    f.close()

stereo_calibrate_from_parameters()
