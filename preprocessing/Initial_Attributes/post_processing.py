import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
import open3d
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


def Obtain_Cluster_Single_PC(cam_points):
    '''make sure the cam points is the torch'''
    cam_points = cam_points.cpu().numpy()
    if len(cam_points)<10:
        return None
    cluster_index = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1).fit_predict(cam_points)

    cam_points = cam_points[cluster_index > -1]
    cluster_index = cluster_index[cluster_index > -1]

    if len(cam_points) < 10:
        return None

    cluster_set = set(cluster_index[cluster_index > -1])
    cluster_sum = np.array([len(cam_points[cluster_index == i]) for i in cluster_set])
    RoI_points = cam_points[cluster_index == np.argmax(cluster_sum)]
    RoI_points = torch.from_numpy(RoI_points)

    return RoI_points

def Convert_Disparity_to_Depth(disparity,baseline=0.60,focal_length=552.5543,mode=1):
    depth = (baseline*focal_length)/(disparity + 1e-6)
    if mode ==1:
        max_value = 192
    if mode==0:
        max_value = 100
    depth = torch.clamp(depth,min=0,max=max_value)
    # depth = np.clip(depth,a_min=0,a_max=100)
    return depth

from sklearn.neighbors import KDTree

def compute_point_density(point_cloud, r):
    # 检查点云是否为空
    if point_cloud.shape[0] == 0:
        return torch.tensor([], device=point_cloud.device)
    
    # 将点云从 torch.Tensor 转换为 numpy 数组
    point_cloud_np = point_cloud.cpu().numpy()
    # 构建 KD 树
    tree = KDTree(point_cloud_np)
    # 查询每个点在半径 r 内的邻居数
    densities = tree.query_radius(point_cloud_np, r, count_only=True) - 1  # 减去自身的点    
    # 转换为 torch.Tensor 并返回
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



