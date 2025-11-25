import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
from vsrd.operations.geometric_operations import project_box_3d
import torchvision
import scipy as sp
from vsrd import utils


LINE_INDICES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]

def decode_box_3d(locations, dimensions, orientations,residual=None):
    # NOTE: use the KITTI-360 "evaluation" format instaed of the KITTI-360 "annotation" format
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/prepare_train_val_windows.py#L133
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L552
    
    # print(locations.shape) #(1,N,3)
    
    if residual is not None:
        locations = locations + residual
    boxes = dimensions.new_tensor([
        [-1.0, -1.0, +1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, +1.0, +1.0],
        [+1.0, +1.0, +1.0],
        [+1.0, +1.0, -1.0],
        [-1.0, +1.0, -1.0],
    ]) * dimensions.unsqueeze(-2) #(1,4,8,3)
    
    boxes = boxes @ orientations.transpose(-2, -1)# make rotation matrix
    boxes = boxes + locations.unsqueeze(-2)
    
    return boxes


def divide_into_n_parts(total, n): 
    if n <= 0:
        raise ValueError("Number of parts must be greater than zero.")
    
    # Calculate the base value for the first N-1 parts
    base_value = total // n
    # Calculate the remainder
    remainder = total % n
    
    # Create a list with N-1 parts of base_value
    parts = [base_value] * (n - 1)
    # Append the last part which is base_value plus the remainder
    parts.append(base_value + remainder)
    
    return parts



# Bilateral Matching: the estimated bounding boxes and GT bounding boxes
def get_dynamic_mask_for_the_world_output(target_inputs,
                                          world_boxes_3d,
                                          dynamic_mask_for_target_view):

    # to the target view camera coordinate
    camera_boxes_3d = torch.einsum("bmn,b...n->b...m", target_inputs.extrinsic_matrices, world_boxes_3d)
    camera_boxes_3d = camera_boxes_3d[..., :-1] / camera_boxes_3d[..., -1:] #(1,4,8,3)---> at source view camera coordinate
    
    # project the 3D bounding boxex from camera coordinate to image coordinate at current  views.
    camera_boxes_2d = torch.stack([
        torch.stack([
            project_box_3d(
                box_3d=camera_box_3d,
                line_indices=LINE_INDICES,
                intrinsic_matrix=intrinsic_matrix,
            )
            for camera_box_3d in camera_boxes_3d
        ], dim=0)
        for camera_boxes_3d, intrinsic_matrix
        in zip(camera_boxes_3d, target_inputs.intrinsic_matrices)
    ], dim=0) #(1,4,2,2)
    
    # make sure inside the image: This is the estimated 2D Bounding Boxes
    camera_boxes_2d = torchvision.ops.clip_boxes_to_image(
        boxes=camera_boxes_2d.flatten(-2, -1),
        size=target_inputs.images.shape[-2:],
    ).unflatten(-1, (2, 2))  #(1,4,2,2)

    # bipartite_matching
    matching_cost_matrices = [
        -torchvision.ops.distance_box_iou(
            boxes1=pd_boxes_2d.flatten(-2, -1),
            boxes2=gt_boxes_2d.flatten(-2, -1),
        )
        for pd_boxes_2d, gt_boxes_2d
        in zip(camera_boxes_2d, target_inputs.boxes_2d)
    ]

    matched_indices = list(map(
        utils.torch_function(sp.optimize.linear_sum_assignment),
        matching_cost_matrices,
    )) 
    
    # (tensor([0, 1, 2, 3]), tensor([1, 0, 2, 3]))]  第一个数组表示 target_outputs 的索引。第二个数组表示 source_inputs 的索引。

    initial_estimated_matched_indices, initial_gt_matched_indices = matched_indices[0]
    initial_estimated_matched_indices_list = initial_estimated_matched_indices.cpu().numpy().tolist() # list for estimated 
    initial_gt_matched_indices_list = initial_gt_matched_indices.cpu().numpy().tolist() # list for GT

    dynamic_mask_for_target_view_for_output = calculate_framewise_dynamic_mask(
                    N=len(initial_gt_matched_indices_list),
                    target_frame_dynamic_list=dynamic_mask_for_target_view,
                    target_frame_matched_indices=initial_gt_matched_indices_list,
                    source_frame_matched_indices=initial_estimated_matched_indices_list)
    
    
    return dynamic_mask_for_target_view_for_output



# FIXME Here: May Existing Bugs
def get_dynamic_sequentail_for_the_world_output(target_inputs,
                                          world_boxes_3d,
                                          target_speed,
                                          target_location):

    # to the target view camera coordinate
    camera_boxes_3d = torch.einsum("bmn,b...n->b...m", target_inputs.extrinsic_matrices, world_boxes_3d)
    camera_boxes_3d = camera_boxes_3d[..., :-1] / camera_boxes_3d[..., -1:] #(1,4,8,3)---> at source view camera coordinate
    
    # project the 3D bounding boxex from camera coordinate to image coordinate at current  views.
    camera_boxes_2d = torch.stack([
        torch.stack([
            project_box_3d(
                box_3d=camera_box_3d,
                line_indices=LINE_INDICES,
                intrinsic_matrix=intrinsic_matrix,
            )
            for camera_box_3d in camera_boxes_3d
        ], dim=0)
        for camera_boxes_3d, intrinsic_matrix
        in zip(camera_boxes_3d, target_inputs.intrinsic_matrices)
    ], dim=0) #(1,4,2,2)
    
    # make sure inside the image: This is the estimated 2D Bounding Boxes
    camera_boxes_2d = torchvision.ops.clip_boxes_to_image(
        boxes=camera_boxes_2d.flatten(-2, -1),
        size=target_inputs.images.shape[-2:],
    ).unflatten(-1, (2, 2))  #(1,4,2,2)

    # bipartite_matching
    matching_cost_matrices = [
        -torchvision.ops.distance_box_iou(
            boxes1=pd_boxes_2d.flatten(-2, -1),
            boxes2=gt_boxes_2d.flatten(-2, -1),
        )
        for pd_boxes_2d, gt_boxes_2d
        in zip(camera_boxes_2d, target_inputs.boxes_2d)
    ]

    matched_indices = list(map(
        utils.torch_function(sp.optimize.linear_sum_assignment),
        matching_cost_matrices,
    )) 
    
    # (tensor([0, 1, 2, 3]), tensor([1, 0, 2, 3]))]  第一个数组表示 target_outputs 的索引。第二个数组表示 source_inputs 的索引。

    initial_estimated_matched_indices, initial_gt_matched_indices = matched_indices[0]
    initial_estimated_matched_indices_list = initial_estimated_matched_indices.cpu().numpy().tolist() # list for estimated 
    initial_gt_matched_indices_list = initial_gt_matched_indices.cpu().numpy().tolist() # list for GT

    
    re_order_velo,re_order_loc = re_order_framewise_dynamic_mask(gt_loc=target_location,gt_velo=target_speed,
                                    target_frame_matched_indices=initial_gt_matched_indices_list,
                                    source_frame_matched_indices=initial_estimated_matched_indices_list)
    
    return  re_order_velo,re_order_loc

def re_order_framewise_dynamic_mask(gt_loc,
                                    gt_velo,

                                    target_frame_matched_indices, 
                                    source_frame_matched_indices):
    
    re_order_loc = torch.zeros_like(gt_loc)
    re_order_velo = torch.zeros_like(gt_velo)

    
    instance_num_0 = re_order_loc.shape[0]
    instance_num_1 = re_order_velo.shape[0]

    assert instance_num_0 == instance_num_1
    
    for i in range(instance_num_0):

        A_index = target_frame_matched_indices[i] # GT index
        B_index = source_frame_matched_indices[i] # estimate index
        
        
        re_order_loc[B_index] = gt_loc[A_index]
        re_order_velo[B_index] = gt_velo[A_index]


        
        
    return re_order_velo,re_order_loc
        
        
    

def calculate_framewise_dynamic_mask(N, target_frame_dynamic_list, target_frame_matched_indices, source_frame_matched_indices):
    # 初始化List B_Mask为全False
    B_Mask = [False] * N
    # 遍历List A的每一个索引
    for i in range(N):
      A_index = target_frame_matched_indices[i]
      B_index = source_frame_matched_indices[i]

      Bool_value = target_frame_dynamic_list[A_index]
      B_Mask[B_index] = Bool_value
      
    return B_Mask