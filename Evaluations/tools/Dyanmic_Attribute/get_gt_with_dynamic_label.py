import os
import json
import glob
import argparse
import functools
import multiprocessing

import tqdm
import torch
import torch.nn as nn
import torchvision
import numpy as np
import pycocotools.mask
import sys
from tqdm import tqdm

def encode_box_3d(boxes_3d):

    locations = torch.mean(boxes_3d, dim=-2)

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

    dimensions = torch.stack([widths, heights, lengths], dim=-1)

    orientations = torch.mean(torch.sub(
        boxes_3d[..., [1, 0, 4, 5], :],
        boxes_3d[..., [2, 3, 7, 6], :],
    ), dim=-2)

    orientations = nn.functional.normalize(orientations[..., [2, 0]], dim=-1)
    orientations = torch.atan2(*reversed(torch.unbind(orientations, dim=-1)))

    return locations, dimensions, orientations

def get_organized_data(annotation_filename,class_names=['car']):

    with open(annotation_filename) as file:
        annotation = json.load(file)
    
    if ('masks' in annotation.keys()) and ("boxes_3d" in annotation.keys()):
        if 'car' in annotation['boxes_3d'].keys():
            
            if len(annotation['boxes_3d']['car'])>0:
                intrinsic_matrix = annotation['intrinsic_matrix']
                extrinsic_matrix = annotation['extrinsic_matrix']

                instance_ids = {
                    class_name: list(masks.keys())
                    for class_name, masks in annotation["masks"].items()
                    if class_name in class_names
                }

                gt_class_names = sum([
                    [class_name] *  len(instance_ids)
                    for class_name, instance_ids in instance_ids.items()
                ], [])
                

                gt_boxes_3d = torch.cat([
                    torch.as_tensor([
                        annotation["boxes_3d"][class_name].get(instance_id, [[np.nan] * 3] * 8)
                        for instance_id in instance_ids
                    ], dtype=torch.float)
                    for class_name, instance_ids in instance_ids.items()
                ], dim=0)


                gt_masks = torch.cat([
                    torch.as_tensor(np.stack([
                        pycocotools.mask.decode(annotation["masks"][class_name][instance_id])
                        for instance_id in instance_ids
                    ]), dtype=torch.float)
                    for class_name, instance_ids in instance_ids.items()
                ], dim=0)

                gt_boxes_2d = torchvision.ops.masks_to_boxes(gt_masks).unflatten(-1, (2, 2))

                instance_ids = instance_ids['car']
            
            
                return instance_ids,gt_boxes_3d,gt_masks,gt_boxes_2d,extrinsic_matrix,intrinsic_matrix
            else:
                return None,None,None,None,None,None

        else:
            return None,None,None,None,None,None
    else:
        return None,None,None,None,None,None

def kept_valid_relative_index(relative_idx,max_lenth):
    kepted_valid_list = []
    
    for data in relative_idx:
        if data>=0 and data<=(max_lenth-1):
            kepted_valid_list.append(data)
    
    return kepted_valid_list

def bounding_box_cam_to_world(bbox_cam, world_to_cam):
    # 添加齐次坐标
    ones = torch.ones((bbox_cam.shape[0], 1), dtype=bbox_cam.dtype, device=bbox_cam.device)
    bbox_cam_homogeneous = torch.cat([bbox_cam, ones], dim=-1)  # [8, 4]

    # 计算 cam_to_world 矩阵
    cam_to_world = torch.inverse(world_to_cam)
    #cam_to_world = world_to_cam

    # 应用 cam_to_world 变换
    bbox_world_homogeneous = bbox_cam_homogeneous @ cam_to_world.T  # [8, 4]

    # 移除齐次坐标，转换回 [8, 3]
    bbox_world = bbox_world_homogeneous[:, :3] / bbox_world_homogeneous[:, 3].unsqueeze(-1)
    
    return bbox_world

def save_prediction(filename, class_names, boxes_3d, boxes_2d, scores,labels):

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        for class_name, box_3d, box_2d, score,label in zip(class_names, boxes_3d, boxes_2d, scores,labels):
            location, dimension, orientation = encode_box_3d(box_3d)

            # ================================================================
            # KITTI-3D definition
            location[..., 1] += dimension[..., 1] / 2.0
            dimension = dimension[..., [1, 0, 2]]
            ray_orientation = torch.atan2(*reversed(torch.unbind(location[..., [2, 0]], dim=-1)))
            global_orientation = orientation - np.pi / 2.0
            local_orientation = global_orientation - ray_orientation
            # ================================================================
            
            file.write(
                f"{class_name.capitalize()} "
                f"{0.0} "
                f"{0} "
                f"{local_orientation} "
                f"{' '.join(map(str, box_2d.flatten().tolist()))} "
                f"{' '.join(map(str, dimension.tolist()))} "
                f"{' '.join(map(str, location.tolist()))} "
                f"{global_orientation} "
                f"{score} "
                f"{label}\n"
            )

def get_neighbor_search_x(current_annotation_name,radius_x=8,threshold=0.01):
    
    current_annotation_folder = current_annotation_name[:-len(os.path.basename(current_annotation_name))]
    current_annotation_filename = os.path.basename(current_annotation_name)
    neighourhood_list = sorted(os.listdir(current_annotation_folder))
    max_length = len(neighourhood_list)
    current_index_in_dir = neighourhood_list.index(current_annotation_filename)
    relative_index_list = list(range(-radius_x,radius_x))
    relative_index_list = [data + current_index_in_dir for data in relative_index_list]
    relative_index_list = kept_valid_relative_index(relative_idx=relative_index_list,max_lenth=max_length)

    returned_folder_list = []
    
    for id in relative_index_list:
        assert os.path.exists(os.path.join(current_annotation_folder,neighourhood_list[id]))
        returned_folder_list.append(os.path.join(current_annotation_folder,neighourhood_list[id]))
        

    neighbour_dict = dict()    
    for idx, fname in enumerate(returned_folder_list):
        source_instance_ids,source_gt_boxes_3d,souce_gt_masks,souce_gt_boxes_2d,source_extrinsic_matrix,souce_intrinsic_matrix = get_organized_data(fname)
        
        if source_instance_ids is not None:
            current_relative_idx = relative_index_list[idx] - current_index_in_dir
            if current_relative_idx not in neighbour_dict.keys():
                neighbour_dict[current_relative_idx] = dict()
                neighbour_dict[current_relative_idx]['boxed_3d'] = source_gt_boxes_3d
                neighbour_dict[current_relative_idx]['boxed_2d'] = souce_gt_boxes_2d
                neighbour_dict[current_relative_idx]['instance_ids'] = source_instance_ids
                neighbour_dict[current_relative_idx]['extrinsic_matrix'] = torch.tensor(source_extrinsic_matrix)
    
    # compute the resiudal between the current view and the source view to decide whether it is dynamic or static
    
    assert 0 in neighbour_dict.keys()
    target_view_dict = neighbour_dict[0]
    target_view_instances = neighbour_dict[0]['instance_ids']
    target_boxes_3d = neighbour_dict[0]['boxed_3d'] #(N,8,3)
    
    target_view_dynamic_list = []
    
    for idx, my_target_instance in enumerate(target_view_instances):
        
        target_box_3d = target_boxes_3d[idx] #[8,3]
        target_box_3d_world = bounding_box_cam_to_world(target_box_3d,target_view_dict['extrinsic_matrix'])
        
        
        target_box_velocity_list = []
        
        for key in neighbour_dict.keys():
            if key==0:
                continue
            source_view_dict = neighbour_dict[key]
            if my_target_instance in source_view_dict['instance_ids']:
                souce_box_3d = source_view_dict['boxed_3d'][source_view_dict['instance_ids'].index(my_target_instance)]
                
                souce_box_3d_world = bounding_box_cam_to_world(souce_box_3d,source_view_dict['extrinsic_matrix'])
                
                velocity = (souce_box_3d_world.mean() - target_box_3d_world.mean()) / key
                
                target_box_velocity_list.append(velocity)
        
        if len(target_box_velocity_list)>0:
            velocity = sum(target_box_velocity_list)/len(target_box_velocity_list)
        else:
            velocity = 0.2
        
        if velocity>=threshold:
            target_view_dynamic_list.append(1.0)
        else:
            target_view_dynamic_list.append(0.0)
    
    
    assert len(target_view_dynamic_list) == target_boxes_3d.shape[0]
    neighbour_dict[0]['dynamic_label'] = target_view_dynamic_list
    
    return neighbour_dict[0]
            
def dynamic_attribute_func(sequence, root_dirname, ckpt_dirname, class_names,json_folder,output_labelname,
                           threshold):
    
    
    prediction_dirname = os.path.join(json_folder, os.path.basename(ckpt_dirname))
    prediction_filenames = sorted(glob.glob(os.path.join(root_dirname, prediction_dirname, sequence, "image_00", "data_rect", "*.json")))
    

    for prediction_filename in tqdm(prediction_filenames):
        
        # Build annotation filename by replacing predictions/{ckpt_basename} with annotations
        # More reliable method: get relative path and replace the folder name
        rel_path = os.path.relpath(prediction_filename, root_dirname)
        ckpt_basename = os.path.basename(ckpt_dirname)
        # Replace predictions/{ckpt_basename} with annotations
        if rel_path.startswith(f"predictions/{ckpt_basename}/"):
            annotation_rel_path = rel_path.replace(f"predictions/{ckpt_basename}/", "annotations/", 1)
        elif rel_path.startswith("predictions/"):
            annotation_rel_path = rel_path.replace("predictions/", "annotations/", 1)
        else:
            # Fallback: try to replace prediction_dirname
            annotation_rel_path = rel_path.replace(prediction_dirname, "annotations")
        annotation_filename = os.path.join(root_dirname, annotation_rel_path)


        instance_ids,gt_boxes_3d,gt_masks,gt_boxes_2d,_,_ = get_organized_data(annotation_filename=annotation_filename,
                                                                               class_names=class_names)
        
        
        if gt_boxes_3d ==None:
            continue


        
        resultd_dict = get_neighbor_search_x(current_annotation_name=annotation_filename,
                                             threshold=threshold)
        
        try:
            assert resultd_dict['instance_ids'] == instance_ids
            assert resultd_dict['boxed_3d'].mean() == gt_boxes_3d.mean()
            assert resultd_dict['boxed_2d'].mean() == gt_boxes_2d.mean()

            dynamic_label_list = resultd_dict['dynamic_label']
            
            gt_class_names = ['car'] * len(dynamic_label_list)
            
            # Build output path: {output_labelname}/{ckpt_basename}/{sequence}/image_00/data_rect/{frame_id}.txt
            # Get relative path from prediction_filename
            rel_path = os.path.relpath(prediction_filename, root_dirname)
            # Remove .json extension
            rel_path_no_ext = os.path.splitext(rel_path)[0]
            # Replace predictions/{ckpt_basename} with {output_labelname}/{ckpt_basename}
            ckpt_basename = os.path.basename(ckpt_dirname)
            if rel_path_no_ext.startswith(f"predictions/{ckpt_basename}/"):
                # Replace: predictions/{ckpt_basename}/sequence/... -> {output_labelname}/{ckpt_basename}/sequence/...
                rel_path_no_ext = rel_path_no_ext.replace(f"predictions/{ckpt_basename}/", f"{output_labelname}/{ckpt_basename}/", 1)
            elif rel_path_no_ext.startswith("predictions/"):
                # Fallback: just replace predictions with output_labelname
                rel_path_no_ext = rel_path_no_ext.replace("predictions/", f"{output_labelname}/", 1)
            label_filename = os.path.join(root_dirname, f"{rel_path_no_ext}.txt")
        
            os.makedirs(os.path.dirname(label_filename), exist_ok=True)

            save_prediction(
                filename=label_filename,
                class_names=gt_class_names,
                boxes_3d=gt_boxes_3d,
                boxes_2d=gt_boxes_2d,
                scores=torch.ones(len(gt_class_names)),
                labels=dynamic_label_list)
        except:
            continue
        
        
        

def main(args):

    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "data_2d_raw", "*")))))
    # dynamic_seqences = [sequences[2],sequences[6]] # for ablation studies
    
    dynamic_seqences = sequences
    
    # dynamic_seqences.remove('2013_05_28_drive_0004_sync')

    with multiprocessing.Pool(args.num_workers) as pool:
        with tqdm(total=len(sequences)) as progress_bar:
            for _ in pool.imap_unordered(functools.partial(
                dynamic_attribute_func,
                root_dirname=args.root_dirname,
                ckpt_dirname=args.ckpt_dirname,
                class_names=args.class_names,
                json_folder = args.json_foldername,
                output_labelname = args.output_labelname,
                threshold=args.dynamic_threshold
            ), dynamic_seqences):

                progress_bar.update(1)
    
    


if __name__=="__main__":
        
    parser = argparse.ArgumentParser(description="Dynamic Attribute")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-360")
    parser.add_argument("--ckpt_dirname", type=str, default="ckpts/kitti_360/vsrd")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--num_workers", type=int, default=9)
    parser.add_argument("--json_foldername", type=str, default=9)
    parser.add_argument("--output_labelname", type=str, default=9)
    parser.add_argument("--dynamic_threshold", type=float, default=0.01)
    args = parser.parse_args()
    
    main(args)
