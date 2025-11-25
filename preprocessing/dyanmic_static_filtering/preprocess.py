import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import skimage.io
import torchvision
import json
import pycocotools.mask
from tqdm import tqdm
sys.path.append("../..")
from PIL import Image

from preprocessing.APIs.optical_flow_estimator import Load_Optical_Flow_Model,InputPadder
from preprocessing.optical_flow_estimation.flow_vis import Tensor_Flow_To_Color
import argparse
import matplotlib.pyplot as plt
import random
import cv2
import multiprocessing
from functools import partial



def read_sampled_image_filenames(filenames):
    image_filenames = [] # save the image list    
    with open(filenames) as file:
        for line in file:
            # 3 colums: 1 is the instance id 
            _, target_image_filename, source_relative_indices = line.strip().split(" ")
            source_relative_indices = list(map(int, source_relative_indices.split(",")))
            image_filenames.append((target_image_filename, source_relative_indices))
    return image_filenames

def get_annotation_filename(image_filename):
    annotation_filename = (
        image_filename
        .replace("data_2d_raw", "annotations")
        .replace(".png", ".json")
    )
    return annotation_filename

def get_image_filename(image_filename, index,image_full_list):
    name = image_full_list[index]
    image_filename = os.path.join(
        os.path.dirname(image_filename),
        name,
    )
    return image_filename

def read_image(image_filename):
    image = skimage.io.imread(image_filename)
    image = torchvision.transforms.functional.to_tensor(image)
    return image

def read_annotation(annotation_filename,class_names =['car']):

    with open(annotation_filename) as file:
        annotation = json.load(file)

    intrinsic_matrix = torch.as_tensor(annotation["intrinsic_matrix"])
    extrinsic_matrix = torch.as_tensor(annotation["extrinsic_matrix"])

    instance_ids = {
        class_name: list(masks.keys())
        for class_name, masks in annotation["masks"].items()
        if class_name in class_names
    }

    if instance_ids:

        masks = torch.cat([
            torch.as_tensor(np.stack([
                pycocotools.mask.decode(annotation["masks"][class_name][instance_id])
                for instance_id in instance_ids
            ]), dtype=torch.float)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0) #(N,H,W)

        labels = torch.cat([
            torch.as_tensor([class_names.index(class_name)] * len(instance_ids), dtype=torch.long)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0) # class

        boxes_3d = torch.cat([
            torch.as_tensor([
                annotation["boxes_3d"][class_name].get(instance_id, [[np.nan] * 3] * 8)
                for instance_id in instance_ids
            ], dtype=torch.float)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        instance_ids = torch.cat([
            torch.as_tensor(list(map(int, instance_ids)), dtype=torch.long)
            for instance_ids in instance_ids.values()
        ], dim=0)

        return dict(
            masks=masks,
            labels=labels,
            boxes_3d=boxes_3d,
            instance_ids=instance_ids,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
        )

    else:

        return dict(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
        )

def get_depth_filename(image_filename):
    # /media/zliu/data12/dataset/KITTI/VSRD_Format/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/
    # /media/zliu/data12/dataset/KITTI/VSRD_Format/pseudo_depth_left/2013_05_28_drive_0000_sync/image_00/data_rect/
    annotation_filename = (
        image_filename
        .replace("data_2d_raw", "pseudo_depth_left")
    )
    return annotation_filename

def read_depth(filename):
    depth = np.array(Image.open(filename))
    depth = depth.astype(np.float32) / 256.
    return depth

def get_target_input(image_filename,class_names=['car']):

    annotation_filename = get_annotation_filename(image_filename)
    image = read_image(image_filename) # get the images
    # get annotations
    annotation = read_annotation(annotation_filename,class_names=class_names)
    
    
    depth =torch.from_numpy(read_depth(get_depth_filename(image_filename))).unsqueeze(0).unsqueeze(0)

    if len(annotation.keys())>2:
        prev_cam_ids = annotation["instance_ids"]
        prev_cam_masks = annotation["masks"]
        
    else:
        return None
    prev_cam_intrinsic_matrix = annotation["intrinsic_matrix"]
    prev_cam_extrinsic_matrix = annotation["extrinsic_matrix"]
    
    
    inputs = dict(
        instance_ids = prev_cam_ids,
        mask = prev_cam_masks,
        intrinsic_matrix = prev_cam_intrinsic_matrix,
        extrinsic_matrix = prev_cam_extrinsic_matrix,
        depth = depth,
        image=image,
        filename=image_filename,
    )
    
    
    return inputs

def neighour_image_filenames(source_name,neighbour_sample,image_filenames_all):
    
    current_index = image_filenames_all.index(os.path.basename(source_name))
    
    prev_frames_nums = neighbour_sample//2
    search_idx = [idx -prev_frames_nums for idx in  list(range(neighbour_sample+1))]
    
    
    search_idx = [idx+current_index for idx in search_idx]
    
    new_search_idx = []
    for ind in search_idx:
        if ind>=0 and ind<len(image_filenames_all)-1:
            new_search_idx.append(ind)
    

    neighour_images_list = [get_image_filename(source_name,idx,image_filenames_all) for idx in new_search_idx]
    
    valid_neighour_image_list = []
    for fname in neighour_images_list:
        if os.path.exists(fname):
            if os.path.exists(get_annotation_filename(fname)):
                valid_neighour_image_list.append(fname)
    
    return valid_neighour_image_list

def compute_top_percent_mean(tensor, percent=0.8):
    non_zero_elements = tensor[tensor != 0]
    if non_zero_elements.numel() == 0:
        return tensor.mean()  
    sorted_elements = torch.sort(non_zero_elements)[0]
    top_percent_length = int(percent * sorted_elements.numel())
    top_percent_elements = sorted_elements[:top_percent_length]
    mean_value = top_percent_elements.mean()


    return mean_value

def warp_image_by_flow(imageB, flow):
    B, C, H, W = imageB.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid_x = grid_x.to(imageB.device).float()
    grid_y = grid_y.to(imageB.device).float()

    # [H, W, 2]
    base_grid = torch.stack((grid_x, grid_y), dim=2)  # [H, W, 2]

    # [B, H, W, 2]
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

    # [B, H, W, 2]
    flow = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]

    # get gid
    sampling_grid = base_grid + flow  # [B, H, W, 2]

    # Normalize into [-1, 1]
    sampling_grid[..., 0] = 2.0 * (sampling_grid[..., 0] / (W - 1)) - 1.0  # x 方向
    sampling_grid[..., 1] = 2.0 * (sampling_grid[..., 1] / (H - 1)) - 1.0  # y 方向

    # Use grid_sample 
    warped_imageB = torch.nn.functional.grid_sample(
        imageB, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    mask = torch.ones_like(imageB)
    valid_mask = F.grid_sample(mask, sampling_grid, mode='bilinear', padding_mode='zeros')
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1

    return warped_imageB,valid_mask

def warp_image_by_depth(imageA, imageB, depthA, K, world2camA, world2camB):
    
    device = imageA.device
    B, C, H, W = imageA.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    y, x = torch.meshgrid(
        torch.arange(0, H, device=device), 
        torch.arange(0, W, device=device), 
        indexing='ij'
    )  # [H, W]
    x = x.expand(B, -1, -1)  # [B, H, W]
    y = y.expand(B, -1, -1)  # [B, H, W]


    depth = depthA[:, 0, :, :]  # [B, H, W]

    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    ones = torch.ones_like(Z)
    points_camA = torch.stack((X, Y, Z, ones), dim=-1)  # [B, H, W, 4]


    cam2worldA = torch.inverse(world2camA).to(device)  # [4, 4]
    points_world = points_camA.reshape(B, -1, 4) @ cam2worldA.T  # [B, H*W, 4]


    points_camB = points_world @ world2camB.T  # [B, H*W, 4]


    X_camB = points_camB[:, :, 0]
    Y_camB = points_camB[:, :, 1]
    Z_camB = points_camB[:, :, 2]

    Z_camB[Z_camB == 0] = 1e-6

    u_proj = (X_camB / Z_camB) * fx + cx
    v_proj = (Y_camB / Z_camB) * fy + cy


    u_norm = 2.0 * (u_proj / (W - 1)) - 1.0
    v_norm = 2.0 * (v_proj / (H - 1)) - 1.0


    sampling_grid = torch.stack((u_norm, v_norm), dim=-1)  # [B, H*W, 2]
    sampling_grid = sampling_grid.reshape(B, H, W, 2)  # [B, H, W, 2]


    valid_mask = (depth > 0).float()
    sampling_grid[valid_mask.unsqueeze(-1).repeat(1,1,1,2) == 0] = 2.0  # 超出 [-1, 1] 的值


    warped_imageB = F.grid_sample(
        imageB, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    mask = torch.ones_like(imageB)
    valid_mask = F.grid_sample(mask, sampling_grid, mode='bilinear', padding_mode='zeros')
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1

    return warped_imageB,valid_mask

def merge_mask_into_whole(mask):
    '''
    input mask shape: [N,H,W]
    
    return: [H,W]
    
    '''
    instance_nums = mask.shape[0]
    returned_mask = torch.zeros((mask.shape[1],mask.shape[2])).type_as(mask)
    
    for idx in range(instance_nums):
        returned_mask = returned_mask + mask[idx]
        returned_mask = torch.clamp(returned_mask,min=0,max=1.0)
    
    return returned_mask.unsqueeze(0).unsqueeze(0)
    
def find_object_flow_using_pertrained(imageB,flow,depthA,K,world2camA,world2camB):
    
    # find the 2D coordinate defined by optical flow and mask

    device = imageB.device
    B, C, H, W = imageB.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid_x = grid_x.to(imageB.device).float()
    grid_y = grid_y.to(imageB.device).float()

    # [H, W, 2]
    base_grid = torch.stack((grid_x, grid_y), dim=2)  # [H, W, 2]
    # [B, H, W, 2]
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
    # [B, H, W, 2]
    flow = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    # get gid
    sampling_grid_defined_by_flow = base_grid + flow  # [B, H, W, 2]

    # Normalize into [-1, 1]
    sampling_grid_defined_by_flow_normalized = torch.zeros_like(sampling_grid_defined_by_flow)
    sampling_grid_defined_by_flow_normalized[..., 0] = 2.0 * (sampling_grid_defined_by_flow[..., 0] / (W - 1)) - 1.0  # x 方向
    sampling_grid_defined_by_flow_normalized[..., 1] = 2.0 * (sampling_grid_defined_by_flow[..., 1] / (H - 1)) - 1.0  # y 方向
    mask_defined_by_flow = torch.ones_like(imageB)
    valid_mask_defined_by_flow = F.grid_sample(mask_defined_by_flow , sampling_grid_defined_by_flow_normalized, mode='bilinear', padding_mode='zeros')
    valid_mask_defined_by_flow[valid_mask_defined_by_flow < 0.9999] = 0
    valid_mask_defined_by_flow[valid_mask_defined_by_flow > 0] = 1
    
    
    # find the 2D coordinate defined by depth warping.
    # 提取相机内参
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    y, x = torch.meshgrid(
        torch.arange(0, H, device=device), 
        torch.arange(0, W, device=device), 
        indexing='ij'
    )  # [H, W]
    x = x.expand(B, -1, -1)  # [B, H, W]
    y = y.expand(B, -1, -1)  # [B, H, W]

    # 获取深度值
    depth = depthA[:, 0, :, :]  # [B, H, W]

    # 计算相机 A 下的三维坐标
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    ones = torch.ones_like(Z)
    points_camA = torch.stack((X, Y, Z, ones), dim=-1)  # [B, H, W, 4]

    # 将点从相机 A 坐标系转换到世界坐标系
    cam2worldA = torch.inverse(world2camA).to(device)  # [4, 4]
    points_world = points_camA.reshape(B, -1, 4) @ cam2worldA.T  # [B, H*W, 4]

    # 将点从世界坐标系转换到相机 B 坐标系
    points_camB = points_world @ world2camB.T  # [B, H*W, 4]

    # 投影到图像平面
    X_camB = points_camB[:, :, 0]
    Y_camB = points_camB[:, :, 1]
    Z_camB = points_camB[:, :, 2]

    # 避免除以零
    Z_camB[Z_camB == 0] = 1e-6

    u_proj = (X_camB / Z_camB) * fx + cx
    v_proj = (Y_camB / Z_camB) * fy + cy

    # 归一化到 [-1, 1]
    # u_norm = 2.0 * (u_proj / (W - 1)) - 1.0
    # v_norm = 2.0 * (v_proj / (H - 1)) - 1.0

    sampling_grid = torch.stack((u_proj, v_proj), dim=-1)  # [B, H*W, 2]
    sampling_grid_defined_by_depth = sampling_grid.reshape(B, H, W, 2)  # [B, H, W, 2]


    sampling_grid_defined_by_depth_normalized = torch.zeros_like(sampling_grid_defined_by_depth)
    sampling_grid_defined_by_depth_normalized[..., 0] = 2.0 * (sampling_grid_defined_by_depth[..., 0] / (W - 1)) - 1.0  # x 方向
    sampling_grid_defined_by_depth_normalized[..., 1] = 2.0 * (sampling_grid_defined_by_depth[..., 1] / (H - 1)) - 1.0  # y 方向
    mask_defined_by_depth = torch.ones_like(imageB)
    valid_mask_defined_by_depth = F.grid_sample(mask_defined_by_depth , sampling_grid_defined_by_depth_normalized, mode='bilinear', padding_mode='zeros')
    valid_mask_defined_by_depth[valid_mask_defined_by_depth < 0.9999] = 0
    valid_mask_defined_by_depth[valid_mask_defined_by_depth > 0] = 1
    
    
    return sampling_grid_defined_by_flow, sampling_grid_defined_by_depth,valid_mask_defined_by_flow,valid_mask_defined_by_depth

def convert_tensor_into_image(tensor_input):
    assert tensor_input.shape[1]==3    
    return tensor_input.squeeze(0).permute(1,2,0).cpu().numpy()

def instance_static_dynamic_accumulation(motion_mask,instance_ids,instance_masks,result_dict,valid_mask):


    instance_masks = shrink_masks_torch(instance_masks)
    assert instance_masks.shape[0] == instance_ids.shape[0]
    
    for idx, current_instance_id  in enumerate(instance_ids):
        instance_mask = instance_masks[idx].unsqueeze(0).unsqueeze(0) # get instance mask
        instance_mask = instance_mask * valid_mask        
        motion_at_this_instance = instance_mask * motion_mask
        
        mean_object_motion = compute_top_percent_mean(motion_at_this_instance,0.8)
        
        masked_areas = instance_mask.sum()
        
        if  current_instance_id.data.item() in result_dict.keys():
            result_dict[current_instance_id.data.item()].append((mean_object_motion,masked_areas))
        else:
            result_dict[current_instance_id.data.item()] = []
            result_dict[current_instance_id.data.item()].append((mean_object_motion,masked_areas))
            
    return result_dict

def shrink_masks_torch(instance_masks, shrink_ratio=0.1):
    N, H, W = instance_masks.shape  # 获取输入形状
    shrunk_masks = torch.zeros_like(instance_masks)  # 创建一个和输入相同大小的 tensor 用于存储结果

    for i in range(N):
        # 将每个实例 mask 从 tensor 转为 numpy 数组
        mask = instance_masks[i].cpu().numpy()

        # 计算核的大小，使其缩小大约 10%
        mask_area = np.sum(mask > 0)  # 计算mask的面积
        erode_kernel_size = int(np.sqrt(mask_area) * shrink_ratio)  # 计算腐蚀核的大小

        if erode_kernel_size > 0:
            kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
            # 使用 cv2.erode 函数进行腐蚀操作
            eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        else:
            eroded_mask = mask  # 如果核大小过小，保留原始mask

        # 将结果转换回 tensor 并存储
        shrunk_masks[i] = torch.tensor(eroded_mask, dtype=instance_masks.dtype)

    return shrunk_masks

def instance_static_dynamic_classifier(result_dict,temperature=100,treshold=0,threshold2=0.33):
    
    '''First Round Selection'''
    initial_processed_dict = dict()
    for keys in result_dict.keys():
        
        if keys not in initial_processed_dict.keys():
            initial_processed_dict[keys] = []
        
        results = result_dict[keys]
        values = torch.stack([items[0] for items in results])
        

        # remove the very small points
        for item in results:
            if item[1]<treshold:
                pass
            else:
                if item[0]>values.mean()*1.5:
                    pass
                else:
                    initial_processed_dict[keys].append(item)
        
        values_valid = (values<1).sum()
        if len(values)>1:
            valid_parts = values_valid/len(values)
        else:
            valid_parts = 1
        
        if valid_parts>threshold2 and len(values)>6:
            for idx, item in enumerate(initial_processed_dict[keys]):
                if item[0]>1.0:
                    initial_processed_dict[keys][idx]=(initial_processed_dict[keys][idx][0]*0+1.5,initial_processed_dict[keys][idx][0])
                    
                    

    return_results = []
    for keys in initial_processed_dict.keys():
        
        results = initial_processed_dict[keys]        
        values = torch.stack([items[0] for items in results])
        weights = F.softmax(torch.stack([torch.sqrt(items[1])/temperature for items in results]))
        mean_2d_variance = sum([weights[i]*values[i] for i in range(len(weights))])
        
        return_results.append(mean_2d_variance)
        

    return return_results

def find_indices(A, B):
    indices = []
    for a in A:
        if a in B:
            indices.append(B.index(a))
        else:
            indices.append(-1)
    return indices       

def search_by_id(old,index):
    new = []
    
    for idx in range(len(index)):
        ind = index[idx]
        new.append(old[ind])
    
    return new

def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic Object Filtering")
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random Seed")

    parser.add_argument(
        "--neighbour_sample", type=int, default=16, help="Random Seed")


    parser.add_argument(
        "--image_folder", type=str, default="/media/zliu/data12/dataset/KITTI/VSRD_Format/data_2d_raw/", help="images folder")


    parser.add_argument(
        "--filename_folder", type=str, default="/media/zliu/data12/dataset/KITTI/VSRD_Format/filenames/R50-N16-M128-B16", help="Random Seed")

    parser.add_argument('--use_multi_thread', 
                        action='store_true', 
                        help="Increase output verbosity")

    parser.add_argument(
        "--data_thread", type=str, default="2013_05_28_drive_0003_sync", help="Random Seed")

    parser.add_argument(
        "--saved_folder", type=str, default="/media/zliu/data12/dataset/KITTI/VSRD_Format/dynamic_static/", help="Random Seed")


    parser.add_argument(
        "--optical_flow_model_path", type=str, 
            default="/home/zliu/Desktop/CVPR2025/VSRD-V2/data_pre_processing/Dynamic_Static_Clss_Flow_based/MeMFlow/ckpts/MemFlowNet_kitti.pth", 
                    help="images folder")

    # get the local rank
    args = parser.parse_args()

    return args



def process_data_thread(data_thread, neighbour_sample, cfg,image_root_template, processed_image_filename_template,optical_processor=None,optical_opt=None):
    saved_contents_all = [] 

    os.makedirs(cfg.saved_folder, exist_ok=True)
    saved_name = os.path.join(cfg.saved_folder, f"{data_thread}_est.txt")

    # 获取目标图像
    processed_image_filename_path = processed_image_filename_template.format(cfg.filename_folder, data_thread)
    sampled_image_filenames = read_sampled_image_filenames(processed_image_filename_path)
    
    image_root = image_root_template.format(cfg.image_folder, data_thread)
    image_filenames_all = sorted(os.listdir(image_root))
    
    for index in tqdm(range(len(sampled_image_filenames))):
        
        target_image_filename, source_relative_indices = sampled_image_filenames[index]

        processed_neighour_image_fnames = neighour_image_filenames(
            source_name=target_image_filename,
            neighbour_sample=neighbour_sample,
            image_filenames_all=image_filenames_all
        )
        processed_neighour_image_fnames = sorted(processed_neighour_image_fnames)


        # current scene results
        
        current_result_dict = dict()
        
        for idx, fname in enumerate(processed_neighour_image_fnames):
            if idx==len(processed_neighour_image_fnames)-1:
                continue
            
            
            prev_frame_inputs = get_target_input(image_filename=processed_neighour_image_fnames[idx])
            next_frame_inputs = get_target_input(image_filename=processed_neighour_image_fnames[idx+1])
            
            if prev_frame_inputs is None or next_frame_inputs is None:
                continue
            
            current_prev_frame_filename = os.path.basename(prev_frame_inputs['filename'])
            
            # previous frame            
            prev_frame_instance_idx = prev_frame_inputs['instance_ids']
            prev_frame_mask = prev_frame_inputs['mask']
            prev_frame_instrinsic_matrix = prev_frame_inputs['intrinsic_matrix']
            prev_frame_extrinsic_matrix = prev_frame_inputs['extrinsic_matrix']
            prev_frame_depth = prev_frame_inputs['depth']
            prev_image_data = prev_frame_inputs['image'].unsqueeze(0)

            # next frame            
            next_frame_instance_idx = next_frame_inputs['instance_ids']
            next_frame_mask = next_frame_inputs['mask']
            next_frame_instrinsic_matrix = next_frame_inputs['intrinsic_matrix']
            next_frame_extrinsic_matrix = next_frame_inputs['extrinsic_matrix']
            next_frame_depth = next_frame_inputs['depth']
            next_image_data = next_frame_inputs['image'].unsqueeze(0)
            
        
    
            padder = InputPadder(prev_image_data.shape)
            prev_image_data = padder.pad(prev_image_data)
            prev_image_data_normalized = prev_image_data * 2.0 -1.0
            prev_image_data_normalized = prev_image_data_normalized.cuda()


            padder = InputPadder(next_image_data.shape)
            next_image_data = padder.pad(next_image_data)
            next_image_data_normalized = next_image_data * 2.0 -1.0
            next_image_data_normalized  = next_image_data_normalized.cuda()
            
            
            # estimate the optical flow
            flow_prev = None
            processed_images_concatenation = torch.cat((prev_image_data_normalized,next_image_data_normalized),dim=0).unsqueeze(0)
            with torch.no_grad():
                flow_low, flow_pre = optical_processor.step(processed_images_concatenation, end=False,
                                            add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)
                flow_pre = padder.unpad(flow_pre[0]).cpu()
                
            estimated_optical_flow = flow_pre.unsqueeze(0).cpu()


            # get warped 2D grid
            warped_image_by_flow, valid_mask = warp_image_by_flow(next_image_data,estimated_optical_flow)
            warped_image_by_flow = warped_image_by_flow * valid_mask
            warped_image_by_depth, valid_mask = warp_image_by_depth(imageA=prev_image_data,imageB=next_image_data,
                                                                    depthA=prev_frame_depth,
                                                                    world2camA=prev_frame_extrinsic_matrix,
                                                                    world2camB=next_frame_extrinsic_matrix,
                                                                    K=next_frame_instrinsic_matrix)
            warped_image_by_depth = warped_image_by_depth * valid_mask
            


            coord_next_by_flow, coord_next_by_depth, valid_mask_0,valid_mask_1 = find_object_flow_using_pertrained(imageB=prev_image_data,
                                            depthA=prev_frame_depth,
                                            world2camA=prev_frame_extrinsic_matrix,
                                            flow=estimated_optical_flow,
                                            world2camB=next_frame_extrinsic_matrix,
                                            K=next_frame_instrinsic_matrix)

            valid_mask_total = valid_mask_0 * valid_mask_1
            valid_mask_total = valid_mask_total[:,:1,:,:].permute(0,2,3,1)
            
    
            coord_next_by_flow_valid = coord_next_by_flow * valid_mask_total
            coord_next_by_depth_valid = coord_next_by_depth * valid_mask_total
                    
            objection_motion = torch.abs((coord_next_by_flow_valid-coord_next_by_depth_valid))
            objection_motion = torch.sum(objection_motion,dim=-1,keepdim=True).permute(0,3,1,2)
            objection_motion = objection_motion * merge_mask_into_whole(prev_frame_mask)
            


            current_result_dict =  instance_static_dynamic_accumulation(motion_mask=objection_motion,
                                                instance_ids=prev_frame_instance_idx,
                                                instance_masks=prev_frame_mask,
                                                result_dict=current_result_dict,
                                                valid_mask=valid_mask_total.permute(0,3,1,2))
            
            

        mean_2d_variance = instance_static_dynamic_classifier(current_result_dict)
        
    
        assert len(mean_2d_variance) == len(current_result_dict.keys())
        
        # prev_frame_instance_idx = prev_frame_instance_idx.cpu().numpy().tolist()
        searched_frame_idx = [int(id) for id in current_result_dict.keys()]
        
        saved_instance_ids = ', '.join([str(key) for key in searched_frame_idx])
        saved_mean_2d_variance = ', '.join(str(data.data.item()) for data in mean_2d_variance)
        saved_lines = saved_instance_ids + " " + target_image_filename + " " + saved_mean_2d_variance
        
        saved_contents_all.append(saved_lines)
        
    
    with open(saved_name,'w') as f:
        for idx, line in enumerate(saved_contents_all):
            if idx!=len(saved_contents_all)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)  





if __name__=="__main__":
    
    
    multiprocessing.set_start_method('spawn', force=True)

    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    

    saved_contents_all = [] 
    neighbour_sample = args.neighbour_sample
        
    os.makedirs(args.saved_folder,exist_ok=True)
    image_root_template = "{}/{}/image_00/data_rect/"
    processed_image_filename_template = "{}/{}/sampled_image_filenames.txt"
    
    
    # get the sceneflow_pretrained_model    
    optical_processor,optical_flow_cfg = Load_Optical_Flow_Model(model_name="MeMFlow",device="cuda:0",model_path=args.optical_flow_model_path)



    if not args.use_multi_thread:
        data_thread = args.data_thread
        saved_name = os.path.join(args.saved_folder,args.data_thread+"_gt.txt") 
        process_data_thread(data_thread=args.data_thread,neighbour_sample=neighbour_sample,
                            cfg=args,
                            image_root_template=image_root_template,
                            processed_image_filename_template=processed_image_filename_template,
                            optical_processor=optical_processor,
                            optical_opt=optical_flow_cfg
                            )
    
    else:
        num_workers = 8 
        
        data_thread_list = os.listdir(args.image_folder)

        # 多线程并行处理每个 data_thread
        with multiprocessing.Pool(num_workers) as pool:
            pool.map(partial(
                process_data_thread,
                neighbour_sample=neighbour_sample,
                cfg=args,
                image_root_template=image_root_template,
                processed_image_filename_template = processed_image_filename_template,
                optical_processor=optical_processor,
                optical_opt=optical_flow_cfg

            ), data_thread_list)
    
  
        
        
            
            
            
            
            
            
            

            

        


    
