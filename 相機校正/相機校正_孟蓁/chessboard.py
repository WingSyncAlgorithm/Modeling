import cv2
import numpy as np
import glob

# 棋盤格的列和行數
chessboard_size = (8, 11)

# 準備棋盤格的三維座標
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 存放棋盤格角點的座標
objpoints = []
imgpoints = []

# 讀取圖片
images = glob.glob('c*.jpg')  # 修改為指定圖片的列表模式

for fname in images: 
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 嘗試找到棋盤格角點
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 繪製角點
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)
    else:
        print(f"Failed to find corners in {fname}")
        # 如果未找到角點，手動創建一個空的 corners 數組
        corners = np.zeros((1, chessboard_size[0] * chessboard_size[1], 2), dtype=np.float32)
        imgpoints.append(corners)
    

cv2.destroyAllWindows()

# 進行相機校正
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 獲取新的相機矩陣和裁剪區域
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, gray.shape[::-1], 1, gray.shape[::-1])

# 儲存校正結果
np.savez('camera_calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# 測試相機校正效果
test_img = cv2.imread('test2.jpg')
h, w = test_img.shape[:2]

# 使用新的相機矩陣進行校正
undistorted_img = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# 裁剪區域
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]

# 顯示原始圖片和校正後的圖片
cv2.imshow('Original Image', test_img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.imwrite('Undistorted Image.jpg', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(ret)
print("\ncamera matrix: ")
print(new_camera_matrix)
print("\ndist coeffs: ")
print(dist_coeffs)
print("\nrvecs: ")
print(rvecs)
print("\ntvecs: ")
print(tvecs)

"""result:
1.2921604905427357

camera matrix:
[[1.08803241e+03 0.00000000e+00 5.29465831e+02]
 [0.00000000e+00 1.10962805e+03 7.57286857e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

dist coeffs:
[[ 0.23633985 -1.07227933 -0.00222332 -0.00375556  1.72323827]]

rvecs:
(array([[-0.57926735],
       [ 0.34451569],
       [ 3.06117561]]), array([[0.28812086],
       [0.26287382],
       [3.07166369]]), array([[-0.46340239],
       [ 0.34968206],
       [ 3.06399319]]), array([[ 0.33461101],
       [-0.35928436],
       [-3.09647785]]), array([[ 0.22501265],
       [-0.38908455],
       [-3.10908084]]), array([[ 0.01948299],
       [-0.37741526],
       [-3.10953834]]), array([[0.66632232],
       [0.22054575],
       [3.0392663 ]]), array([[0.53326674],
       [0.24825032],
       [3.06159694]]), array([[0.45777053],
       [0.24335126],
       [3.04642473]]), array([[0.37184901],
       [0.2578417 ],
       [3.05642415]]))

tvecs:
(array([[ 4.34119544],
       [ 4.96779603],
       [17.70144021]]), array([[ 1.16117725],
       [ 5.4757448 ],
       [15.87988175]]), array([[ 3.510695  ],
       [ 4.97878283],
       [17.31604785]]), array([[ 3.32881196],
       [ 5.20706043],
       [16.71275385]]), array([[ 2.74960717],
       [ 4.96474381],
       [16.65618969]]), array([[ 2.4989657 ],
       [ 5.45695485],
       [16.33112943]]), array([[-0.13023627],
       [ 5.87965546],
       [15.62160051]]), array([[-0.22865934],
       [15.809808  ]]), array([[ 0.35618159],
       [ 5.58839032],
       [15.72793063]]), array([[ 0.90107975],
       [ 5.50218879],
       [15.77748836]]))
"""
