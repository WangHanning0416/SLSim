import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

intrinsics = np.array([[297.6375381033778,0.0,255.5,0.00],
                       [0.0,297.6375381033778,255.5,0.00],
                       [0.0 ,0.0             ,1.0  ,0.00],
                       [0.00,0.00            ,0.00 ,1.00]])

cam2proj = np.array([[1.00, 0.00, 0.00, 0.20],
                    [0.00, 1.00, 0.00, 0.00],
                    [0.00, 0.00, 1.00, -0.10],
                    [0.00, 0.00, 0.00, 1.00]])

def depth_to_point_cloud(depth, intrinsics):
    h,w = depth.shape
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)
    Z = depth
    X = (x - intrinsics[0, 2]) * Z / intrinsics[0, 0]
    Y = (y - intrinsics[1, 2]) * Z / intrinsics[1, 1]
    point_cloud = np.stack((X, Y, Z), axis=-1)  #(H, W, 3)
    return point_cloud

def transform_point_cloud(point_cloud, cam2proj):
    h, w, _ = point_cloud.shape
    pc_flat = point_cloud.reshape(-1, 3)
    pc_hom = np.hstack([pc_flat, np.ones((pc_flat.shape[0], 1))])  # (N, 4)
    proj_points_hom = (cam2proj @ pc_hom.T).T  # (N, 4)
    proj_points = proj_points_hom[:, :3] / proj_points_hom[:, 3:4]
    return proj_points.reshape(h, w, 3)

def project_to_pattern_plane(proj_points, K_proj):
    X = proj_points[:, :, 0]
    Y = proj_points[:, :, 1]
    Z = proj_points[:, :, 2]
    Z[Z == 0] = 1e-6

    fx, fy = K_proj[0, 0], K_proj[1, 1]
    cx, cy = K_proj[0, 2], K_proj[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return u, v


def sample_pattern(u, v, pattern):
    h, w = u.shape
    u_wrapped = np.mod(np.round(u).astype(np.int32), pattern.shape[1])
    v_wrapped = np.mod(np.round(v).astype(np.int32), pattern.shape[0])

    sampled = np.zeros((h, w, 3), dtype=np.uint8)

    if pattern.ndim == 2:
        sampled[:, :, 0] = pattern[v_wrapped, u_wrapped]
        sampled[:, :, 1] = pattern[v_wrapped, u_wrapped]
        sampled[:, :, 2] = pattern[v_wrapped, u_wrapped]
    elif pattern.ndim == 3 and pattern.shape[2] == 3:
        for c in range(3):
            sampled[:, :, c] = pattern[v_wrapped, u_wrapped, c]
    else:
        raise ValueError("Unsupported pattern shape")

    return sampled

def apply_depth_attenuation(image, depth, Z0=0.8, gamma=1.0, min_val=0.2):
    attenuation = (Z0 / np.clip(depth, 0.1, 5.0)) ** gamma
    attenuation = np.clip(attenuation, min_val, 1.0)
    attenuation = attenuation[:, :, np.newaxis]
    return (image.astype(np.float32) * attenuation).clip(0, 255).astype(np.uint8)

def main():
    path1 = ".//Data//rgb//0000003.jpg"
    image = cv2.imread(path1)#(512, 512, 3)
    path2 = ".//Data//depth//0000003.png"
    depth = cv2.imread(path2,cv2.IMREAD_UNCHANGED)#(512, 512)
    path3 = ".//Data//pattern//alacarte.png"
    pattern = cv2.imread(path3, cv2.IMREAD_UNCHANGED)#(1280,800)
    point_cloud = depth_to_point_cloud(depth, intrinsics)

    depth = depth.astype(np.float32) / 1000.0  # Convert depth to meters

    point_cloud = depth_to_point_cloud(depth, intrinsics)  # shape: (h, w, 3)
    proj_points = transform_point_cloud(point_cloud, cam2proj)  # shape: (h, w, 3)
    K_proj = intrinsics[:3, :3]
    u, v = project_to_pattern_plane(proj_points, K_proj)
    projected_image = sample_pattern(u, v, pattern)  # shape (H, W, 3)
    projected_image = apply_depth_attenuation(projected_image, proj_points[:, :, 2])

    rgb_img = image.copy()
    overlay_img = cv2.addWeighted(rgb_img, 0.8, projected_image, 0.5, 0)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.title("RGB with Structured Light Projection")
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    cv2.imwrite(".//generated//alacarteSim.png", overlay_img)

if __name__ == "__main__":
    main()
