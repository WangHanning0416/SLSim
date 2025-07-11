import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

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

def apply_depth_blur(image, depth, focus=1.0, strength=3):
    norm_depth = (depth - depth.min()) / (depth.max() - depth.min())
    blur_image = cv2.GaussianBlur(image, (5,5), strength)
    alpha = np.clip((norm_depth - focus) * 5, 0, 1)[:,:,np.newaxis]
    return (image * (1 - alpha) + blur_image * alpha).astype(np.uint8)

def simulate_occlusion_blur(image, mask_ratio=0.05):
    h, w, _ = image.shape
    num_masks = np.random.randint(5, 20)
    for _ in range(num_masks):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2 = min(w, x1 + np.random.randint(10, 50))
        y2 = min(h, y1 + np.random.randint(10, 50))
        if np.random.rand() < 0.5:
            # mask out
            image[y1:y2, x1:x2] = 0
        else:
            # blur
            patch = image[y1:y2, x1:x2]
            image[y1:y2, x1:x2] = cv2.GaussianBlur(patch, (5,5), 3)
    return image

def visualize_pattern_usage(u, v, pattern_shape):
    """
    显示哪些 pattern 图案区域在投影过程中被使用了

    参数：
        u, v: 反投影后 pattern 图像平面上的像素坐标（通常为浮点）
        pattern_shape: pattern 图像的形状 (H, W)
    """
    h_pat, w_pat = pattern_shape[:2]
    
    # 将坐标转换为整数索引，并裁剪到合法范围
    u_idx = np.round(u).astype(np.int32)
    v_idx = np.round(v).astype(np.int32)

    valid_mask = (u_idx >= 0) & (u_idx < w_pat) & (v_idx >= 0) & (v_idx < h_pat)

    usage_map = np.zeros((h_pat, w_pat), dtype=np.uint8)
    usage_map[v_idx[valid_mask], u_idx[valid_mask]] = 255

    plt.figure(figsize=(8, 6))
    plt.title("Pattern Usage Map")
    plt.imshow(usage_map, cmap='gray')
    plt.xlabel("Pattern Width")
    plt.ylabel("Pattern Height")
    plt.colorbar(label='Used (255) vs Unused (0)')
    plt.tight_layout()
    plt.show()

    return usage_map

def main():
    path1 = ".//Data//rgb//0000003.jpg"
    image = cv2.imread(path1)#(512, 512, 3)
    path2 = ".//Data//depth//0000003.png"
    depth = cv2.imread(path2,cv2.IMREAD_UNCHANGED)#(512, 512)
    path3 = ".//Data//pattern//alacarte.png"
    pattern = cv2.imread(path3, cv2.IMREAD_UNCHANGED)#(1280,800)
    intrinsics = np.array([[297.6375381033778,0.0,255.5,0.00],
                            [0.0,297.6375381033778,255.5,0.00],
                            [0.0 ,0.0             ,1.0  ,0.00],
                            [0.00,0.00            ,0.00 ,1.00]])
    point_cloud = depth_to_point_cloud(depth, intrinsics)

    depth = depth.astype(np.float32) / 1000.0  # Convert depth to meters

    point_cloud = depth_to_point_cloud(depth, intrinsics)  # shape: (h, w, 3)
    proj_points = transform_point_cloud(point_cloud, cam2proj)  # shape: (h, w, 3)
    K_proj = intrinsics = np.array([[680,0.0,590,0.00],
                                    [0.0,380,350,0.00],
                                    [0.0 ,0.0,1.0  ,0.00],
                                    [0.00,0.00,0.00 ,1.00]])
    u, v = project_to_pattern_plane(proj_points, K_proj)
    # usage_map = visualize_pattern_usage(u, v, pattern.shape)

    projected_image = sample_pattern(u, v, pattern)  # shape (H, W, 3)
    projected_image = apply_depth_attenuation(projected_image, proj_points[:, :, 2])
    #projected_image = apply_depth_blur(projected_image, proj_points[:, :, 2], focus=depth.min()+0.1, strength=3)
    #projected_image = simulate_occlusion_blur(projected_image, mask_ratio=0.05)

    rgb_img = image.copy()
    overlay_img = cv2.addWeighted(rgb_img, 0.65, projected_image, 0.5, 0)

    cv2.imwrite(".//generated//alacarte_origin.png", overlay_img)

if __name__ == "__main__":
    main()
