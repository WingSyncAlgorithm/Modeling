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
