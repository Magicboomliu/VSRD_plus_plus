import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
import cv2
import open3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree


def cluster_point_cloud(cam_points):
    """DBSCAN cluster filter; keep the largest cluster."""
    cam_points = cam_points.cpu().numpy()
    if len(cam_points) < 10:
        return None

    cluster_index = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1).fit_predict(cam_points)
    cam_points = cam_points[cluster_index > -1]
    cluster_index = cluster_index[cluster_index > -1]

    if len(cam_points) < 10:
        return None

    cluster_ids = set(cluster_index[cluster_index > -1])
    cluster_sizes = np.array([len(cam_points[cluster_index == cid]) for cid in cluster_ids])
    roi_points = cam_points[cluster_index == list(cluster_ids)[np.argmax(cluster_sizes)]]
    return torch.from_numpy(roi_points)


Obtain_Cluster_Single_PC = cluster_point_cloud


def compute_point_density(point_cloud, radius):
    """Count neighbors within ``radius`` for each point."""
    if point_cloud.shape[0] == 0:
        return torch.tensor([], device=point_cloud.device)

    point_cloud_np = point_cloud.cpu().numpy()
    tree = KDTree(point_cloud_np)
    densities = tree.query_radius(point_cloud_np, radius, count_only=True) - 1
    return torch.tensor(densities, device=point_cloud.device)



def update_instance_masks(warp_error, instance_masks,ratio=0.0):

    B, N, H, W = instance_masks.shape  
    warp_error_flat = warp_error.view(B, 1, H * W)  
    instance_masks_flat = instance_masks.view(B, N, H * W)     
    updated_masks_flat = torch.zeros_like(instance_masks_flat) 

    for i in range(N):
        instance_mask = instance_masks_flat[:, i, :] 
        instance_mask = instance_mask.unsqueeze(1)
        instance_warp_error = warp_error_flat[instance_mask == 1].view(-1) 
        if instance_warp_error.numel() > 0: 
            percentile_value = torch.quantile(instance_warp_error, ratio)
            retain_positions = (warp_error_flat <= percentile_value).float().expand_as(instance_mask)
            updated_masks_flat[:, i, :] = instance_mask * retain_positions


    updated_instance_masks = updated_masks_flat.view(B, N, H, W)
    return updated_instance_masks


def shrink_masks_torch(instance_masks, shrink_ratio=0.1):
    """Erode instance masks by a fraction of sqrt(mask area)."""
    N, H, W = instance_masks.shape
    shrunk_masks = torch.zeros_like(instance_masks)

    for i in range(N):
        mask = instance_masks[i].cpu().numpy()
        mask_area = np.sum(mask > 0)
        erode_kernel_size = int(np.sqrt(mask_area) * shrink_ratio)

        if erode_kernel_size > 0:
            kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
            eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        else:
            eroded_mask = mask

        shrunk_masks[i] = torch.tensor(eroded_mask, dtype=instance_masks.dtype)

    return shrunk_masks


def icp_translation_only(A, B, max_iterations=100, tolerance=1e-6):
    """
    使用 ICP 进行点云的配准，仅考虑平移，不考虑旋转。
    
    Args:
    - A: 点云 A，形状为 (N1, 3) 的 numpy 数组
    - B: 点云 B，形状为 (N2, 3) 的 numpy 数组
    - max_iterations: 最大迭代次数
    - tolerance: 收敛阈值
    
    Returns:
    - translation: 最优的平移向量 (1, 3)
    - transformed_A: 应用平移后的 A 点云
    - residual: 平均残差，形状为 (1, 3)
    """
    translation = np.zeros(3)
    kdtree = cKDTree(B)

    for i in range(max_iterations):
        transformed_A = A + translation
        distances, indices = kdtree.query(transformed_A)
        
        B_nearest = B[indices]
        centroid_A = np.mean(transformed_A, axis=0)
        centroid_B = np.mean(B_nearest, axis=0)
        
        new_translation = centroid_B - centroid_A
        if np.linalg.norm(new_translation - translation) < tolerance:
            break
        
        translation += new_translation

    transformed_A = A + translation
    residual = np.mean(transformed_A - B[indices], axis=0)
    
    translation = translation.reshape(1,3)

    return translation


def get_orientation_from_point_cloud(pcd):
    if pcd is None:
        return np.pi / 2
    if len(pcd)<2:
        return np.pi/2

    depth_points_np_xz = pcd[:, [0, 2]]

    '''orient'''
    # orient_set = [(i[1] - j[1]) / (i[0] - j[0]) for j in depth_points_np_xz
    #                 for i in depth_points_np_xz]
    
    orient_set = []
    for j_idx, j in enumerate(depth_points_np_xz):
        for i_idx, i in enumerate(depth_points_np_xz):
            if i_idx != j_idx: # 避免选择同一对点
                if i[0] != j[0]: # 避免除以零错误
                    orient = (i[1] - j[1]) / (i[0] - j[0])
                    orient_set.append(orient)
        
    orient_sort = np.array(sorted(np.array(orient_set).reshape(-1)))
    orient_sort = np.arctan(orient_sort[~np.isnan(orient_sort)])
    orient_sort_round = np.around(orient_sort, decimals=1)
    set_orenit = list(set(orient_sort_round))

    ind = np.argmax([np.sum(orient_sort_round == i) for i in set_orenit])
    orient = set_orenit[ind]
    if orient < 0:
        orient += np.pi

    if orient > np.pi / 2 + np.pi * 3 / 8:
        orient -= np.pi / 2
    if orient < np.pi / 8:
        orient += np.pi / 2

    if np.max(pcd[:, 0]) - np.min(pcd[:, 0]) > 4 and \
            (orient >= np.pi / 8 and orient <= np.pi / 2 + np.pi * 3 / 8):
        if orient < np.pi / 2:
            orient += np.pi / 2
        else:
            orient -= np.pi / 2
    

    return orient



