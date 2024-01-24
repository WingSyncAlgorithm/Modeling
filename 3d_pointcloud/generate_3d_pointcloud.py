import cv2
import numpy as np
import open3d as o3d
from PIL import Image

def show_point_cloud(pointcloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pointcloud)
    vis.run()
    vis.destroy_window()

def main():
    f = 718.856
    cx = 607.1928
    cy = 185.2157
    b = 0.573

    disparity_sgbm = np.asarray(Image.open('sgbm_disparity.jpg'))  # 視差圖
    left_img = np.asarray(Image.open('left.png'))  # 左相機圖
    disparity = np.float32(disparity_sgbm) / 16.0

    valid_points = disparity > 10
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector([
        [(u - cx) / f * disparity[v, u], (v - cy) / f * disparity[v, u], (f * b / disparity[v, u])*0.2]
        for v in range(disparity.shape[0])
        for u in range(disparity.shape[1])
        if valid_points[v, u]
    ])
    pointcloud.colors = o3d.utility.Vector3dVector([
        left_img[v, u] / 255.0 for v in range(disparity.shape[0]) for u in range(disparity.shape[1]) if valid_points[v, u]
    ])

    show_point_cloud(pointcloud)

    cv2.imshow("disparity", disparity / 96)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()