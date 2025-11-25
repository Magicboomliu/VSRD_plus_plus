import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch

def safe_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # 形状检查
    assert a.dim() == 3 and b.dim() == 3, f"Expect 3D tensors, got {a.shape}, {b.shape}"
    assert a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1], \
        f"Batch/inner dims mismatch: {a.shape} x {b.shape}"

    # dtype / device / contiguous 统一
    a = a.to(dtype=torch.float32, memory_format=torch.contiguous_format).contiguous()
    b = b.to(dtype=torch.float32, memory_format=torch.contiguous_format).contiguous()

    # 快速“自检”——有些版本 cuBLAS 遇到 NaN/Inf 也会直接 INVALID_VALUE
    if torch.isnan(a).any() or torch.isinf(a).any() or torch.isnan(b).any() or torch.isinf(b).any():
        raise ValueError("Input contains NaN/Inf; clean your tensors before bmm.")

    # 路径1：einsum（很多情况下可避开 cuBLAS 的触发点）
    try:
        return torch.einsum('bij,bjk->bik', a, b)
    except RuntimeError:
        pass

    # 路径2：matmul
    try:
        return a @ b
    except RuntimeError:
        pass

    # 路径3：baddbmm（有时调用的是不同的代码路径）
    try:
        out = torch.zeros((a.shape[0], a.shape[1], b.shape[2]), device=a.device, dtype=a.dtype)
        return torch.baddbmm(out, a, b, beta=0.0, alpha=1.0)
    except RuntimeError:
        pass

    # 路径4：最后兜底，逐 batch 计算（慢，但稳）
    outs = []
    for i in range(a.shape[0]):
        outs.append(a[i].mm(b[i]))
    return torch.stack(outs, dim=0)




def transform_bounding_boxes_to_world(extrinsic_matrix, bounding_boxes_camera):
    '''
    Inputs: 
    (1) Extrinsic Matrix: Cam2Word
    (2) Bounding Boxes: 3D bounding boxes from the camera prespective
    '''
    N = bounding_boxes_camera.shape[0]
    ones = torch.ones((N, 8, 1), device=bounding_boxes_camera.device)  # [N, 8, 1]
    bounding_boxes_camera_homogeneous = torch.cat([bounding_boxes_camera, ones], dim=-1)  # [N, 8, 4]

    extrinsic_matrix = extrinsic_matrix.expand(N, -1, -1)  # [N, 4, 4]   
    

     
    bounding_boxes_world_homogeneous = torch.bmm(extrinsic_matrix, bounding_boxes_camera_homogeneous.transpose(1, 2)).transpose(1, 2)  # [N, 8, 4]

    bounding_boxes_world = bounding_boxes_world_homogeneous[..., :3] / bounding_boxes_world_homogeneous[..., 3:4]
    
    return bounding_boxes_world


def rotation_matrix_y(cos, sin):
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)
    rotation_matrices = torch.stack([
        torch.stack([ cos, zero,  sin], dim=-1),
        torch.stack([zero,  one, zero], dim=-1),
        torch.stack([-sin, zero,  cos], dim=-1),
    ], dim=-2)
    return rotation_matrices

def encode_box_3d(boxes_3d):
    # (N,4,3)
    locations = torch.mean(boxes_3d, dim=-2) #[1,N,3]

    widths = torch.mean(torch.norm(torch.sub(
        boxes_3d[..., [1, 2, 6, 5], :],
        boxes_3d[..., [0, 3, 7, 4], :],
    ), dim=-1), dim=-1)

    heights = torch.mean(torch.norm(torch.sub(
        boxes_3d[..., [4, 5, 6, 7], :],
        boxes_3d[..., [0, 1, 2, 3], :],
    ), dim=-1), dim=-1)

    lengths = torch.mean(torch.norm(torch.sub(
        boxes_3d[..., [1, 0, 4, 5], :],
        boxes_3d[..., [2, 3, 7, 6], :],
    ), dim=-1), dim=-1)

    dimensions = torch.stack([widths, heights, lengths], dim=-1) / 2.0

    orientations = torch.mean(torch.sub(
        boxes_3d[..., [1, 0, 4, 5], :],
        boxes_3d[..., [2, 3, 7, 6], :],
    ), dim=-2) # speed
    

    orientations = nn.functional.normalize(orientations[..., [2, 0]], dim=-1) # x,y

    orientations = rotation_matrix_y(*torch.unbind(orientations, dim=-1))

    return locations, dimensions, orientations

def generate_point_cloud(image, intrinsics, extrinsics, depth_map):
    
    B, C, H, W = image.shape
    device = image.device
    
    # 1. generated image grid (u, v)
    u = torch.arange(W, device=image.device).view(1, -1).expand(H, W).to(device)
    v = torch.arange(H, device=image.device).view(-1, 1).expand(H, W).to(device)
    ones = torch.ones_like(u).to(device)
    
    # Concatenate (u, v) into pixel homogeneous coordinates [u, v, 1]
    pixel_coords = torch.stack([u, v, ones], dim=0).float()  # Shape: [3, H, W]
    pixel_coords = pixel_coords.unsqueeze(0)  # Shape: [1, 3, H, W]

    # 2. Project pixel coordinates to the camera coordinate system (using camera intrinsics)
    depth_map = depth_map.squeeze(1)  # Remove the channel dimension in the depth map, Shape: [1, H, W]
    pixel_coords = pixel_coords * depth_map  # Multiply by depth to get depth-corrected pixel coordinates

    intrinsics_inv = torch.inverse(intrinsics)  # 
    cam_coords = torch.matmul(intrinsics_inv, pixel_coords.view(B, 3, -1))  # 
    cam_coords = cam_coords.view(B, 3, H, W)  # recover shape to [1, 3, H, W]

    # 3. Convert the point in the camera coordinate system to the world coordinate system (using camera external parameters)
    cam_coords_homogeneous = torch.cat([cam_coords, ones.unsqueeze(0).unsqueeze(0)], dim=1)  # Shape: [1, 4, H, W]
    world_coords = torch.matmul(torch.inverse(extrinsics), cam_coords_homogeneous.view(B, 4, -1))  # 应用外参矩阵
    world_coords = world_coords.view(B, 4, H, W)  # recover shape to [1, 4, H, W]

    # 4. Convert world coordinate homogeneous coordinates to non-homogeneous coordinates (divide by the last dimension of the homogeneous coordinates)
    world_coords = world_coords[:, :3, :, :] / world_coords[:, 3:4, :, :]  # Shape: [1, 3, H, W]

    return world_coords

def match_instance_masks_with_iou(instance_masks, bounding_boxes):
    """
    使用 IoU 匹配，将 instance_masks 从 [N1, H, W] 映射为 [N2, H, W]。
    
    Args:
    - instance_masks (torch.Tensor): 大小为 [N1, H, W] 的实例掩码。
    - bounding_boxes (torch.Tensor): 大小为 [N2, 2, 2] 的 2D 边界框，格式为 [[x_min, y_min], [x_max, y_max]]。
    
    Returns:
    - matched_masks (torch.Tensor): 匹配后的实例掩码，大小为 [N2, H, W]。
    """
    
    def compute_iou(box1, box2):
        """
        计算两个 2D 边界框的 IoU (Intersection over Union)。
        
        Args:
        - box1 (torch.Tensor): 形状为 [2, 2] 的 bounding box，格式为 [[x_min, y_min], [x_max, y_max]]。
        - box2 (torch.Tensor): 形状为 [2, 2] 的 bounding box，格式为 [[x_min, y_min], [x_max, y_max]]。
        
        Returns:
        - iou (float): IoU 值。
        """
        # 计算交集
        x_min = torch.max(box1[0, 0], box2[0, 0])
        y_min = torch.max(box1[0, 1], box2[0, 1])
        x_max = torch.min(box1[1, 0], box2[1, 0])
        y_max = torch.min(box1[1, 1], box2[1, 1])
        
        inter_area = torch.clamp(x_max - x_min, min=0) * torch.clamp(y_max - y_min, min=0)
        
        # 计算并集
        box1_area = (box1[1, 0] - box1[0, 0]) * (box1[1, 1] - box1[0, 1])
        box2_area = (box2[1, 0] - box2[0, 0]) * (box2[1, 1] - box2[0, 1])
        
        union_area = box1_area + box2_area - inter_area
        
        # 计算 IoU
        iou = inter_area / union_area
        return iou

    def extract_bounding_box(mask):
        """
        从 instance mask 中提取 2D bounding box。
        
        Args:
        - mask (torch.Tensor): 大小为 [H, W] 的掩码。
        
        Returns:
        - bounding_box (torch.Tensor): 大小为 [2, 2]，格式为 [[x_min, y_min], [x_max, y_max]]。
        """
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        
        if torch.any(rows) and torch.any(cols):
            y_min, y_max = torch.where(rows)[0][[0, -1]]
            x_min, x_max = torch.where(cols)[0][[0, -1]]
        else:
            # 如果 mask 是全 0，则返回一个无效的框
            x_min, x_max, y_min, y_max = 0, 0, 0, 0
        
        return torch.tensor([[x_min, y_min], [x_max, y_max]])

    N1, H, W = instance_masks.shape
    N2 = bounding_boxes.shape[0]
    
    # 存储匹配后的掩码
    matched_masks = torch.zeros((N2, H, W), dtype=instance_masks.dtype).to(instance_masks.device)
    
    # 提取每个 instance mask 的 bounding box
    instance_bounding_boxes = torch.stack([extract_bounding_box(instance_masks[i]) for i in range(N1)])
    
    # 计算 IoU 并匹配
    for j in range(N2):
        best_iou = -1
        best_mask = None
        for i in range(N1):
            iou = compute_iou(instance_bounding_boxes[i], bounding_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_mask = instance_masks[i]
        
        # 匹配最佳 IoU 的掩码
        matched_masks[j] = best_mask
    
    return matched_masks

def Convert_Disparity_to_Depth(disparity,baseline=0.60,focal_length=552.5543,mode=1):
    depth = (baseline*focal_length)/(disparity + 1e-6)
    if mode ==1:
        max_value = 192
    if mode==0:
        max_value = 100
        
    depth = torch.clamp(depth,min=0,max=max_value)
    # depth = np.clip(depth,a_min=0,a_max=100)
    return depth

def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid

def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid

# use disp to warp the left to right
def disp_warp(img, disp, padding_mode='border'):
    """Warping by disparity
    Args:
        img: [B, 3, H, W]
        disp: [B, 1, H, W], positive
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 3, H, W]
        valid_mask: [B, 3, H, W]
    """
    assert disp.min() >= 0

    grid = meshgrid(img)  # [B, 2, H, W] in image scale
    # Note that -disp here
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    warped_img = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode=padding_mode,align_corners=False)

    mask = torch.ones_like(img)
    valid_mask = F.grid_sample(mask, sample_grid, mode='bilinear', padding_mode='zeros',align_corners=False)
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1
    return warped_img, valid_mask

def point_cloud_to_2d_mask(point_cloud, K, world_to_camera, H, W):
    """
    将 3D 点云投影到 2D 图像平面，生成 2D mask。
    
    Args:
    - point_cloud (torch.Tensor): 3D 点云，大小为 [N, 3]。
    - K (torch.Tensor): 相机内参矩阵，大小为 [1, 3, 3]。
    - world_to_camera (torch.Tensor): 相机外参矩阵，大小为 [1, 4, 4]，world to camera。
    - H (int): 图像的高度。
    - W (int): 图像的宽度。
    
    Returns:
    - mask (torch.Tensor): 生成的 2D mask，大小为 [1, 1, H, W]。
    """
    # 1. 将 3D 点云从世界坐标系转换到相机坐标系
    N = point_cloud.shape[0]
    
    # 为点云添加齐次坐标
    ones = torch.ones((N, 1), device=point_cloud.device)
    point_cloud_homogeneous = torch.cat([point_cloud, ones], dim=-1)  # [N, 4]
    
    # 相机外参矩阵 world_to_camera，应用到点云上，转换到相机坐标系
    point_cloud_camera = torch.matmul(world_to_camera[0], point_cloud_homogeneous.t()).t()  # [N, 4]
    point_cloud_camera = point_cloud_camera[:, :3]  # 去掉齐次坐标部分，变成 [N, 3]
    
    # 2. 使用相机内参将 3D 点投影到 2D 图像平面
    point_cloud_2d_homogeneous = torch.matmul(K[0], point_cloud_camera.t()).t()  # [N, 3]
    
    # 将齐次坐标归一化
    u = point_cloud_2d_homogeneous[:, 0] / point_cloud_2d_homogeneous[:, 2]  # 归一化 x 坐标
    v = point_cloud_2d_homogeneous[:, 1] / point_cloud_2d_homogeneous[:, 2]  # 归一化 y 坐标
    
    # 3. 生成 2D mask
    mask = torch.zeros((1, 1, H, W), device=point_cloud.device)  # 创建全 0 的 mask
    
    # 只保留投影在图像内的点
    valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_valid = u[valid_mask].long()
    v_valid = v[valid_mask].long()
    
    # 将投影点对应的 mask 位置设置为 1
    mask[0, 0, v_valid, u_valid] = 1.0
    
    return mask


def remove_largest_and_smallest_point_clouds(tuples_list):
    sizes = [pc.shape[0] for pc, _ in tuples_list]
    max_index = sizes.index(max(sizes))
    min_index = sizes.index(min(sizes))
    filtered_list = [tpl for i, tpl in enumerate(tuples_list) if i != max_index and i != min_index]

    return filtered_list

def remove_point_cloud_with_largest_avg_z(tuples_list):
    avg_z_values = [pc[:, 2].mean().item() for pc, _ in tuples_list]
    max_z_index = avg_z_values.index(max(avg_z_values))
    filtered_list = [tpl for i, tpl in enumerate(tuples_list) if i != max_z_index]

    return filtered_list

def Sort_valid_ROI_LiDAR_by_relative_id(tuples_list):
    sorted_list = sorted(tuples_list, key=lambda x: x[1])
    
    return sorted_list

def rotation_matrix_y_single(cos, sin):
    """
    返回绕 y 轴的 3x3 旋转矩阵，支持 float 类型的 cos 和 sin。
    
    Args:
    - cos (float): 旋转角度的余弦值。
    - sin (float): 旋转角度的正弦值。
    
    Returns:
    - rotation_matrix (torch.Tensor): 3x3 的旋转矩阵。
    """
    # 如果 cos 和 sin 是 float，将它们转换为 torch.Tensor
    cos = torch.tensor(cos) if isinstance(cos, float) else cos
    sin = torch.tensor(sin) if isinstance(sin, float) else sin

    one = torch.tensor(1.0)
    zero = torch.tensor(0.0)

    # 构建 3x3 旋转矩阵
    rotation_matrices = torch.stack([
        torch.stack([ cos, zero,  sin], dim=-1),
        torch.stack([zero,  one, zero], dim=-1),
        torch.stack([-sin, zero,  cos], dim=-1),
    ], dim=-2)

    return rotation_matrices

