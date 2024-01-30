import numpy as np
import cv2 as cv
import glob

def calibrate_stereo_camera():
    """立體視覺標定
    
    輸入:
        left資料夾: 左相機的棋盤格jpg檔
        right資料夾: 右相機的棋盤格jpg檔
    
    輸出:
        camera_parameters.txt: 包含 camera matrix, distortion coefficients matrix, rotation matrix,
                               translation vector, essential matrix, fundamental matrix
    """

    CHECKER_BOARD = (6, 9)  # 棋盤格內角點(格子長、寬各自減一)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    object_points = np.zeros((CHECKER_BOARD[0] * CHECKER_BOARD[1], 3),
                             np.float32)
    object_points[:, :2] = np.mgrid[0:CHECKER_BOARD[0],
                                    0:CHECKER_BOARD[1]].T.reshape(-1, 2)

    real_world_points = []  # 儲存世界座標系角點
    img_points_left = []
    img_points_right = []

    images_left_all = sorted(glob.glob('left\\*.jpg'))
    images_right_all = sorted(glob.glob('right\\*.jpg'))

    for image_left_one, image_right_one in zip(images_left_all,
                                               images_right_all):

        img_left = cv.imread(image_left_one)
        img_right = cv.imread(image_right_one)
        gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

        # 棋盤角點檢測
        return_find_corner_left, corners_left = cv.findChessboardCorners(
            gray_left, CHECKER_BOARD, None)
        return_find_corner_right, corners_right = cv.findChessboardCorners(
            gray_right, CHECKER_BOARD, None)

        if return_find_corner_left and return_find_corner_right:
            real_world_points.append(object_points)

            corners_left_extract = cv.cornerSubPix(gray_left, corners_left,
                                                   (11, 11), (-1, -1),
                                                   criteria)
            img_points_left.append(corners_left_extract)

            corners_right_extract = cv.cornerSubPix(gray_right, corners_right,
                                                    (11, 11), (-1, -1),
                                                    criteria)
            img_points_right.append(corners_right_extract)

        cv.drawChessboardCorners(img_left, CHECKER_BOARD, corners_left_extract,
                                 return_find_corner_left)
        cv.imshow('img left', img_left)
        cv.drawChessboardCorners(img_right, CHECKER_BOARD,
                                 corners_right_extract,
                                 return_find_corner_right)
        cv.imshow('img right', img_right)
        cv.waitKey(500)

    cv.destroyAllWindows()

    # 左右各自標定
    return_calibration_left, camera_matrix_left, distortion_coefficients_left, rotaion_vector_left, translation_vector_left = cv.calibrateCamera(
        real_world_points, img_points_left, gray_left.shape[::-1], None, None)
    height_left, width_left, channels_left = img_left.shape
    new_camera_matrix_left, roi_left = cv.getOptimalNewCameraMatrix(
        camera_matrix_left, distortion_coefficients_left,
        (width_left, height_left), 1, (width_left, height_left))

    return_calibration_right, camera_matrix_right, distortion_coefficients_right, rotaion_vector_right, translation_vector_right = cv.calibrateCamera(
        real_world_points, img_points_right, gray_right.shape[::-1], None,
        None)
    height_right, width_right, channels_right = img_right.shape
    new_camera_matrix_right, roi_right = cv.getOptimalNewCameraMatrix(
        camera_matrix_right, distortion_coefficients_right,
        (width_right, height_right), 1, (width_right, height_right))

    # 雙目標定
    flags = 0
    flags = cv.CALIB_FIX_INTRINSIC

    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100,
                       0.0001)

    return_stereo, new_camera_matrix_left, new_distortion_coefficients_left, new_camera_matrix_right, new_distortion_coefficients_right, R, T, E, F = cv.stereoCalibrate(
        real_world_points, img_points_left, img_points_right,
        camera_matrix_left, distortion_coefficients_left, camera_matrix_right,
        distortion_coefficients_right, gray_left.shape[::-1], criteria_stereo,
        flags)

    # 寫入相機參數到camera_parameters.txt
    f = open("camera_parameters.txt", 'w')

    print(return_stereo, file=f)
    print("\nCamera matrix left: ", file=f)
    print(new_camera_matrix_left, file=f)
    print("\ndistCoeffs left: ", file=f)
    print(new_distortion_coefficients_left, file=f)
    print("\nCamera matrix right: ", file=f)
    print(new_camera_matrix_right, file=f)
    print("\ndistCoeffs right: ", file=f)
    print(new_distortion_coefficients_right, file=f)
    print("\nR: ", file=f)
    print(R, file=f)
    print("\nT: ", file=f)
    print(T, file=f)
    print("\nE: ", file=f)
    print(E, file=f)
    print("\nF: ", file=f)
    print(F, file=f)

    f.close()
    
calibrate_stereo_camera()