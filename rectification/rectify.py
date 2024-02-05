import cv2
import numpy as np


def rectify_images(camera_matrix_path, left_image_path, right_image_path):
    """相片校正

    Args:
        camera_matrix_path: 相機參數.txt的路徑
        left_image_path: 左圖的路徑
        right_image_path: 右圖的路徑

    Returns:
        rect_left_image: 校正後的左圖
        rect_right_image: 校正後的右圖

    """

    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    def read_matrices_from_txt(file_path):
        """讀取相機參數.txt裡面的矩陣

            Arg:
                file_path: txt檔的路徑

            Return:
                matrices: 包含了矩陣標題以及矩陣的字典
        """
        matrices = {}
        current_matrix_lines = []

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                if line.endswith(':'):
                    current_title = line[:-1]  # 把標題後的冒號刪除
                    current_matrix_lines = []
                elif line.endswith(']'):
                    current_matrix_lines.append(line)
                    # 提取出數字，並轉換成 float
                    row_data = list(
                        map(float, ' '.join(current_matrix_lines).strip(
                            '[]').split()))
                    matrices[current_title] = matrices.get(current_title,
                                                           []) + [row_data]
                    current_matrix_lines = []
                else:
                    # 當行結尾不是':'或']'的時候，先把行存起來，等到下一次再合併，解決同一行換行問題
                    current_matrix_lines.append(line)

        # 把 matrices 中矩陣的型態從 list 轉換成 numpy array
        matrices = {key: np.array(value) for key, value in matrices.items()}
        return matrices

    camera_matrix = read_matrices_from_txt(camera_matrix_path)
    camera_matrix_left = camera_matrix.get('Camera matrix left')
    distcoeffs_left = camera_matrix.get('distCoeffs left')
    camera_matrix_right = camera_matrix.get('Camera matrix right')
    distcoeffs_right = camera_matrix.get('distCoeffs right')
    r = camera_matrix.get('R')
    t = camera_matrix.get('T')

    # 計算旋轉矩陣和投影矩陣
    (rotation_left, rotation_right, projection_left, projection_right, Q,
     validPixROI1,
     validPixROI2) = cv2.stereoRectify(camera_matrix_left, distcoeffs_left,
                                       camera_matrix_right, distcoeffs_right,
                                       left_image.shape[1::-1], r, t)

    # 產生左圖映射表(計算無畸變、修正轉換關係)
    (map1,
     map2) = cv2.initUndistortRectifyMap(camera_matrix_left, distcoeffs_left,
                                         rotation_left, projection_left,
                                         left_image.shape[1::-1], cv2.CV_32FC1)
    # 執行映射(將一張影像中某位置的像素放置到另一個圖片指定位置)
    rect_left_image = cv2.remap(left_image, map1, map2, cv2.INTER_CUBIC)

    # 產生右圖映射表
    (map1,
     map2) = cv2.initUndistortRectifyMap(camera_matrix_right, distcoeffs_right,
                                         rotation_right, projection_right,
                                         left_image.shape[1::-1], cv2.CV_32FC1)
    rect_right_image = cv2.remap(right_image, map1, map2, cv2.INTER_CUBIC)

    print(left_image.shape[1::-1])

    return rect_left_image, rect_right_image


rect_left_image, rect_right_image = rectify_images(
    'camera_parameters.txt', 'left10.jpg', 'right10.jpg')

# Save the rectified images
cv2.imwrite('rectified_left.jpg', rect_left_image)
cv2.imwrite('rectified_right.jpg', rect_right_image)
