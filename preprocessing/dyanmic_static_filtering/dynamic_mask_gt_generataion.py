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
import matplotlib.pyplot as plt
import random
import argparse
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
        prev_cam_boxes = annotation["boxes_3d"]
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
        boxes_3d = prev_cam_boxes,
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

def transform_bboxes_to_world(bboxes, world2cam):
    # Convert bounding boxes to homogeneous coordinates
    N, num_points, _ = bboxes.shape
    ones = torch.ones((N, num_points, 1), device=bboxes.device)
    bboxes_homogeneous = torch.cat([bboxes, ones], dim=-1)  # Shape (N, 8, 4)
    
    # Inverse of the world2cam matrix (camera to world)
    cam_to_world = torch.inverse(world2cam)  # Shape (4, 4)
    
    # Transform bounding boxes to world coordinates
    bboxes_world_homogeneous = torch.einsum('ij,nkj->nki', cam_to_world, bboxes_homogeneous)  # Shape (N, 8, 4)
    
    # Convert back to Cartesian coordinates by dropping the homogeneous coordinate
    bboxes_world = bboxes_world_homogeneous[..., :3]  # Shape (N, 8, 3)
    
    return bboxes_world

def find_indices(A, B):
    indices = []
    for a in A:
        if a in B:
            indices.append(B.index(a))
        else:
            indices.append(-1)
    return indices

def record_norm_error(prev_1,next_1,visible_list,prev_instance_ids,dict):
    
    assert len(prev_instance_ids) == prev_1.shape[0]
    for idx,instance_idx in enumerate(prev_instance_ids):
        
        if visible_list[idx]:
            prev_3d = prev_1[idx]
            next_3d = next_1[idx]
            
            error = torch.abs(prev_3d-next_3d)
            velocity = torch.pow(torch.sum(error*error)/3,1/2)

            if instance_idx in dict.keys():
                dict[instance_idx].append(velocity)
            else:
                dict[instance_idx] = []
                dict[instance_idx].append(velocity)
                        
def get_static_dynamic_not(results_dict,threshold=0.2):
    
    return_dict = dict()
    for key in results_dict.keys():
        current_velocity = sum([velo.data.item() for velo in results_dict[key]])/len(results_dict[key])
        is_dynamic = current_velocity>=threshold
        return_dict[key] = float(is_dynamic)

    return return_dict

def search_by_id(old,index):
    new = []
    
    for idx in range(len(index)):
        ind = index[idx]
        new.append(old[ind])
    
    return new

def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic Object Filtering: Using the 3D Boudning Boxes")
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

    # get the local rank
    args = parser.parse_args()

    return args
    
def process_data_thread(data_thread, neighbour_sample, cfg,image_root_template, processed_image_filename_template):
    saved_contents_all = [] 

    os.makedirs(cfg.saved_folder, exist_ok=True)
    saved_name = os.path.join(cfg.saved_folder, f"{data_thread}_gt.txt")

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

        current_result_dict = dict()
        
        for idx, fname in enumerate(processed_neighour_image_fnames):
            if idx == len(processed_neighour_image_fnames) - 1:
                continue
            
            prev_frame_inputs = get_target_input(image_filename=processed_neighour_image_fnames[idx])
            next_frame_inputs = get_target_input(image_filename=processed_neighour_image_fnames[idx+1])
            
            if prev_frame_inputs is None or next_frame_inputs is None:
                continue

            # 获取前后帧信息并计算误差
            prev_frame_boxes3d_world = transform_bboxes_to_world(prev_frame_inputs['boxes_3d'], prev_frame_inputs['extrinsic_matrix'])
            next_frame_boxes3d_world = transform_bboxes_to_world(next_frame_inputs['boxes_3d'], next_frame_inputs['extrinsic_matrix'])
            
            prev_frame_instance_idx = [data.data.item() for data in prev_frame_inputs['instance_ids']]
            next_frame_instance_idx = [data.data.item() for data in next_frame_inputs['instance_ids']]

            next_relative_for_prev = find_indices(prev_frame_instance_idx, next_frame_instance_idx)
            visible_mask = [float(item > -1) for item in next_relative_for_prev]
            next_frame_boxes3d_world_selected = next_frame_boxes3d_world[next_relative_for_prev]

            # 记录速度误差
            record_norm_error(
                prev_1=prev_frame_boxes3d_world.mean(dim=1),
                next_1=next_frame_boxes3d_world_selected.mean(dim=1),
                visible_list=visible_mask,
                prev_instance_ids=prev_frame_instance_idx,
                dict=current_result_dict
            )
        
        static_dynamic_dict = get_static_dynamic_not(results_dict=current_result_dict)
        mean_2d_variance = [data for data in static_dynamic_dict.values()]
        searched_frame_idx = [int(id) for id in static_dynamic_dict.keys()]
        
        saved_instance_ids = ', '.join([str(key) for key in searched_frame_idx])
        saved_mean_2d_variance = ', '.join(str(data) for data in mean_2d_variance)
        saved_lines = saved_instance_ids + " " + target_image_filename + " " + saved_mean_2d_variance
        saved_contents_all.append(saved_lines)

    with open(saved_name, 'w') as f:
        for idx, line in enumerate(saved_contents_all):
            if idx != len(saved_contents_all) - 1:
                f.writelines(line + "\n")
            else:
                f.writelines(line)




if __name__=="__main__":
    
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
    
    if not args.use_multi_thread:
        data_thread = args.data_thread
        saved_name = os.path.join(args.saved_folder,args.data_thread+"_gt.txt") 
        process_data_thread(data_thread=args.data_thread,neighbour_sample=neighbour_sample,
                            cfg=args,
                            image_root_template=image_root_template,
                            processed_image_filename_template=processed_image_filename_template
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

            ), data_thread_list)
