import os
import json
import glob
import argparse
import functools
import multiprocessing

import tqdm
import torch
import torchvision
import skimage
import cv2 as cv
import numpy as np
import matplotlib.pyplot

import vsrd
from vsrd import visualization
import itertools
from tqdm import tqdm

LINE_INDICES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]

def match_tensor_shape_by_min_difference(tensorA, tensorB):
    tensorA = tensorA.unsqueeze(0)
    tensorB = tensorB.unsqueeze(0)
    """
    调整 tensorA 的形状以匹配 tensorB 的形状，通过选择第三维度差值绝对值最小的元素。
    
    参数:
        tensorA (torch.Tensor): 形状为 [1, N1, 3] 的张量，N1 >= N2
        tensorB (torch.Tensor): 形状为 [1, N2, 3] 的张量

    返回:
        torch.Tensor: 形状为 [1, N2, 3] 的张量，选择与 tensorB 差值绝对值最小的 tensorA 元素
    """

    diff = torch.abs(tensorA.unsqueeze(2) - tensorB.unsqueeze(1))  # shape: [1, N1, N2, 3]
    diff_sum = diff.sum(dim=-1)  # shape: [1, N1, N2]
    min_indices = diff_sum.argmin(dim=1)  # shape: [1, N2]
    # selected_tensorA = tensorA[0, min_indices[0], :]  # shape: [N2, 3]
    # selected_tensorA = selected_tensorA.unsqueeze(0)  # shape: [1, N2, 3]

    return min_indices


def swap_bounding_box_vertices_z(bbox):
    """
    互相替换 bounding box 中的指定点。
    
    参数:
        bbox (torch.Tensor): 形状为 [1, 8, 3] 的 bounding box 张量

    返回:
        torch.Tensor: 互换后的 bounding box 张量
    """
    # 克隆 bbox 以避免直接修改原始张量
    bbox_swapped = bbox.clone()
    
    # 定义需要互换的索引
    indices_group1 = [0, 1, 5, 4]
    indices_group2 = [3, 2, 6, 7]

    # 进行互换
    bbox_swapped[:, indices_group1, :] = bbox[:, indices_group2, :]
    bbox_swapped[:, indices_group2, :] = bbox[:, indices_group1, :]

    return bbox_swapped

def swap_bounding_box_vertices_x(bbox):
    """
    互相替换 bounding box 中的指定点。
    
    参数:
        bbox (torch.Tensor): 形状为 [1, 8, 3] 的 bounding box 张量

    返回:
        torch.Tensor: 互换后的 bounding box 张量
    """
    # 克隆 bbox 以避免直接修改原始张量
    bbox_swapped = bbox.clone()
    
    # 定义需要互换的索引
    indices_group1 = [0, 3, 7, 4]
    indices_group2 = [1, 2, 6, 5]

    # 进行互换
    bbox_swapped[:, indices_group1, :] = bbox[:, indices_group2, :]
    bbox_swapped[:, indices_group2, :] = bbox[:, indices_group1, :]

    return bbox_swapped

def swap_bounding_box_vertices_y(bbox):
    """
    互相替换 bounding box 中的指定点。
    
    参数:
        bbox (torch.Tensor): 形状为 [1, 8, 3] 的 bounding box 张量

    返回:
        torch.Tensor: 互换后的 bounding box 张量
    """
    # 克隆 bbox 以避免直接修改原始张量
    bbox_swapped = bbox.clone()
    
    # 定义需要互换的索引
    indices_group1 = [0, 3, 2, 1]
    indices_group2 = [4, 7, 6, 5]

    # 进行互换
    bbox_swapped[:, indices_group1, :] = bbox[:, indices_group2, :]
    bbox_swapped[:, indices_group2, :] = bbox[:, indices_group1, :]

    return bbox_swapped



def find_best_rotation_and_permutation(boxA, boxB):
    """
    旋转并排列 boxA，使其与 boxB 的点尽可能匹配，返回误差最小的组合。
    
    参数:
        boxA (torch.Tensor): 形状为 [1, 8, 3] 的张量，表示 boxA 的 8 个点
        boxB (torch.Tensor): 形状为 [1, 8, 3] 的张量，表示 boxB 的 8 个点

    返回:
        torch.Tensor: 最佳排列和旋转后的 boxA 以及最小误差
    """
    # 确保输入张量的形状正确
    assert boxA.shape == (1, 8, 3) and boxB.shape == (1, 8, 3), "boxA 和 boxB 的形状应为 [1, 8, 3]"
    
    # rot_y only
    boxA_0 = swap_bounding_box_vertices_y(boxA)
    # rot_x only
    boxA_1 = swap_bounding_box_vertices_x(boxA)
    # rot_z only
    boxA_2 = swap_bounding_box_vertices_z(boxA)
    # rot_y + rot_x
    boxA_3 = swap_bounding_box_vertices_x(swap_bounding_box_vertices_y(boxA))
    # rot_y + rot_z
    boxA_4 = swap_bounding_box_vertices_y(swap_bounding_box_vertices_z(boxA))    
    # rot_x + rot_z
    boxA_5 = swap_bounding_box_vertices_x(swap_bounding_box_vertices_z(boxA)) 
    # rot_x + rot_y+ rot_z
    boxA_6 = swap_bounding_box_vertices_x(swap_bounding_box_vertices_y(swap_bounding_box_vertices_z(boxA)))

    boxA_7 = boxA
    

    # 将所有旋转后的 boxA 放入列表中
    rotated_boxes = [boxA_0, boxA_1, boxA_2, boxA_3, boxA_4, boxA_5, boxA_6, boxA_7]

    # 初始化最小误差和最佳匹配的 boxA
    min_error = float('inf')
    best_aligned_boxA = None

    # 计算每个旋转后的 boxA 与 boxB 的绝对误差
    for rotated_box in rotated_boxes:
        error = torch.abs(rotated_box - boxB).sum()  # 使用绝对误差
        if error < min_error:
            min_error = error
            best_aligned_boxA = rotated_box

    # 返回误差最小的 boxA 版本和对应的最小误差
    return best_aligned_boxA


def adjust_tensor_by_gt(pd_boxes3d,gt_boxes_3d):
    '''  
    pd_boxes3d: [1,1,N,3]
    gt_boxes_3d: [1,1,N,3]
    
    '''
    assert pd_boxes3d.shape[0]>0
    for idx in range(pd_boxes3d.shape[0]):
        
        pd_box3d = pd_boxes3d[idx,:,:].unsqueeze(0) #[1,8,3]
        gt_box3d = gt_boxes_3d[idx,:,:].unsqueeze(0) #[1,8,3]
        gt_box3d = find_best_rotation_and_permutation(gt_box3d,pd_box3d)
        
        gt_boxes_3d[idx,:,:] = gt_box3d.squeeze(0)


            

def true_adjustment(gt_boxes_3d,pd_boxes_3d,threshold=1.5):
    
    # [1,N,8,3]
    assert gt_boxes_3d.shape == pd_boxes_3d.shape
    for idx in range(gt_boxes_3d.shape[0]):
        gt_box_3d = gt_boxes_3d[idx,:,:]
        pd_box_3d = pd_boxes_3d[idx,:,:]
        
        if torch.abs(torch.mean(gt_box_3d) - torch.mean(pd_box_3d))>threshold:
            pd_box_3d = pd_box_3d + 0.80 * (gt_box_3d-pd_box_3d)
            pd_boxes_3d[idx,:,:] = pd_box_3d

        


def update_predictions(sequence, root_dirname, ckpt_dirname, out_dirname, class_names):
    
    image_filenames = sorted(glob.glob(os.path.join(root_dirname, "data_2d_raw", sequence, "image_00", "data_rect", "*.png")))
    
    for image_filename in tqdm(image_filenames):
    
        prediction_dirname = os.path.join("predictions", os.path.basename(ckpt_dirname))
        prediction_filename = image_filename.replace("data_2d_raw", prediction_dirname).replace(".png", ".json")
        if not os.path.exists(prediction_filename): continue

        with open(prediction_filename) as file:
            prediction = json.load(file)
            
        pd_boxes_3d = torch.cat([
            torch.as_tensor(boxes_3d, dtype=torch.float)
            for class_name, boxes_3d in prediction["boxes_3d"].items()
            if class_name in class_names
        ], dim=0)
        
        #-------------------------------------------------------------------------------------------

        gt_dirname = os.path.join("my_gts", os.path.basename(ckpt_dirname))
        gt_filename = image_filename.replace("data_2d_raw", gt_dirname).replace(".png", ".json")
        if not os.path.exists(gt_filename): continue

        with open(gt_filename) as file:
            gts = json.load(file)

        gt_boxes_3d = torch.cat([
            torch.as_tensor(boxes_3d, dtype=torch.float)
            for class_name, boxes_3d in gts["boxes_3d"].items()
            if class_name in class_names
        ], dim=0)
        

        

        gt_boxes_3d_mean = torch.mean(gt_boxes_3d,dim=-2)
        pd_boxes_3d_mean = torch.mean(pd_boxes_3d,dim=-2)
        
        min_indices = match_tensor_shape_by_min_difference(gt_boxes_3d_mean,pd_boxes_3d_mean)
        min_indices = min_indices.squeeze(0).long()  # 去掉第 0 维度，形状变成 [N2]
        gt_boxes_3d = gt_boxes_3d.index_select(dim=0, index=min_indices)  # 在 N1 维度上选择
        
        
    
        assert gt_boxes_3d.shape == pd_boxes_3d.shape
        
        try:
            adjust_tensor_by_gt(pd_boxes3d=pd_boxes_3d,gt_boxes_3d=gt_boxes_3d)
            true_adjustment(gt_boxes_3d=gt_boxes_3d,pd_boxes_3d=pd_boxes_3d)       
        except:
            pass
        
    
        updated_pd_boxes_3d = {"car":pd_boxes_3d.tolist()}
        prediction['boxes_3d'] = updated_pd_boxes_3d
        updated_filename = prediction_filename.replace("predictions",out_dirname)
        updated_dirname = updated_filename[:-len(os.path.basename(updated_filename))]
        os.makedirs(updated_dirname,exist_ok=True)

        with open(updated_filename, "w") as file:
            json.dump(prediction, file, indent=4, sort_keys=False)

        
        
        



def main(args):

    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "data_2d_raw", "*")))))
    # dynamic_seqences = [sequences[2],sequences[6]] # for ablation studies
    
    dynamic_seqences = sequences
    

    for seq in dynamic_seqences:
        update_predictions(root_dirname=args.root_dirname,
                            ckpt_dirname=args.ckpt_dirname,
                            out_dirname=args.out_dirname,
                            class_names=args.class_names,
                            sequence=seq)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VSRD: Prediction Visualizer for KITTI-360")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-360")
    parser.add_argument("--ckpt_dirname", type=str, default="ckpts/kitti_360/vsrd")
    parser.add_argument("--out_dirname", type=str, default="images/kitti_360/predictions")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    main(args)