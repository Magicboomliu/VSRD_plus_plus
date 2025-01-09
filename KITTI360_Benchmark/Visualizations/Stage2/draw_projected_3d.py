import numpy as np
import cv2 as cv
import os
import random
import warnings
import pickle
from tqdm import tqdm
import torch
import torch.utils.data as data
import math
import torch.nn.functional as F
import pickle
from PIL import Image
import json
import cv2
import matplotlib.pyplot as plt
import pycocotools.mask
import open3d as o3d
from kitti_box_computation import compute_box_3d_pure


def KITTI36O_Orientation_TO_KITTI3D_Orienat(kitti360_orient):
    kitti3d = kitti360_orient-math.pi/2
    if kitti3d<=-math.pi:
        kitti3d = kitti3d+ math.pi*2
    return kitti3d


def rotation_matrix_y(angles):
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    one = torch.ones_like(angles)
    zero = torch.zeros_like(angles)
    rotation_matrices = torch.stack([
        torch.stack([ cos, zero,  sin], dim=-1),
        torch.stack([zero,  one, zero], dim=-1),
        torch.stack([-sin, zero,  cos], dim=-1),
    ], dim=-2)
    return rotation_matrices

def decode_box_3d(locations, dimensions, orientations):
    # NOTE: use the KITTI-360 "evaluation" format instaed of the KITTI-360 "annotation" format
    # NOTE: the KITTI-360 "annotation" format is different from the KITTI-360 "evaluation" format
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/prepare_train_val_windows.py#L133
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L552
    boxes = dimensions.new_tensor([
        [-1.0, -1.0, +1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, +1.0, +1.0],
        [+1.0, +1.0, +1.0],
        [+1.0, +1.0, -1.0],
        [-1.0, +1.0, -1.0],
    ]) * dimensions.unsqueeze(-2)
    boxes = boxes @ orientations.transpose(-2, -1)
    boxes = boxes + locations.unsqueeze(-2)
    return boxes



def load_roi_lidar(path):
    with open(path, 'rb') as f:
        RoI_box_points = pickle.load(f)
    
    return RoI_box_points 

def read_annotation(annotation_filename,class_names =["N/A",
                    "car"]):
    with open(annotation_filename) as file:
        annotation = json.load(file)
    intrinsic_matrix = torch.as_tensor(annotation["intrinsic_matrix"])
    extrinsic_matrix = torch.as_tensor(annotation["extrinsic_matrix"])
    instance_ids = {
        class_name: list(masks.keys())
        for class_name, masks in annotation["masks"].items()
        if class_name in class_names
    }

    # if contains the instances ids
    if instance_ids:
        masks = torch.cat([
            torch.stack([
                torch.as_tensor(
                    data=pycocotools.mask.decode(annotation["masks"][class_name][instance_id]),
                    dtype=torch.float,
                )
                for instance_id in instance_ids
            ], dim=0)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        labels = torch.cat([
            torch.as_tensor(
                data=[class_names.index(class_name)] *  len(instance_ids),
                dtype=torch.long,
            )
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        boxes_3d = torch.cat([
            torch.stack([
                torch.as_tensor(
                    data=annotation["boxes_3d"][class_name].get(instance_id, [[np.nan] * 3] * 8),
                    dtype=torch.float,
                )
                for instance_id in instance_ids
            ], dim=0)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        instance_ids = torch.cat([
            torch.as_tensor(
                data=list(map(int, instance_ids)),
                dtype=torch.long,
            )
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
    
    # else returan 
    else:
        return dict(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
        )

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def rotation_matrix_y(angles):
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    one = torch.ones_like(angles)
    zero = torch.zeros_like(angles)
    rotation_matrices = torch.stack([
        torch.stack([ cos, zero,  sin], dim=-1),
        torch.stack([zero,  one, zero], dim=-1),
        torch.stack([-sin, zero,  cos], dim=-1),
    ], dim=-2)
    return rotation_matrices

def decode_box_3d(locations, dimensions, orientations):
    # NOTE: use the KITTI-360 "evaluation" format instaed of the KITTI-360 "annotation" format
    # NOTE: the KITTI-360 "annotation" format is different from the KITTI-360 "evaluation" format
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/prepare_train_val_windows.py#L133
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L552
    boxes = dimensions.new_tensor([
        [-1.0, -1.0, +1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, +1.0, +1.0],
        [+1.0, +1.0, +1.0],
        [+1.0, +1.0, -1.0],
        [-1.0, +1.0, -1.0],
    ]) * dimensions.unsqueeze(-2)
    boxes = boxes @ orientations.transpose(-2, -1)
    boxes = boxes + locations.unsqueeze(-2)
    return boxes


def draw3d_bbox_2d_projection_V2(image, qs, color=(255, 0, 0), thickness=1, is_gt=False):
    """
    Draws the projected 3D bounding box on a 2D image.
    If the bounding box is too large, returns np.nan.

    Parameters:
        image (np.ndarray): The input image.
        qs (np.ndarray): Projected 3D bounding box points of shape (8, 2).
        color (tuple): Color for the box lines.
        thickness (int): Line thickness.
        is_gt (bool): If the bounding box is ground truth.

    Returns:
        np.ndarray or np.nan: Image with the bounding box drawn or np.nan if bounding box is too large.
    """
    if is_gt:
        color = (255, 0, 0)
        thickness = 2
    else:
        color = (0, 0, 255)
        thickness = 2

    # Check for box dimensions
    qs = qs.astype(np.int32)
    x_min, y_min = np.min(qs, axis=0)
    x_max, y_max = np.max(qs, axis=0)
    width = x_max - x_min
    height = y_max - y_min

    # Threshold for bounding box size (customize these values as needed)
    image_height, image_width = image.shape[:2]
    max_width = image_width * 0.4
    max_height = image_height * 0.4

    if width > max_width or height > max_height:
        print("Bounding box is too large, returning NaN")
        return np.nan

    # Draw lines if bounding box is valid
    lines = [
        (0, 1), (4, 5), (0, 4), (1, 5), # Front to back edges
        (0, 3), (4, 7), (1, 2), (5, 6), # Top and bottom edges
        (3, 2), (6, 7), (3, 7), (2, 6)  # Side edges
    ]

    for start, end in lines:
        cv2.line(image, tuple(qs[start]), tuple(qs[end]), color, thickness, lineType=cv2.LINE_AA)

    return image



def draw3d_bbox_2d_projection(image, qs, color=(255, 0, 0), thickness=1,is_gt=False):
    
    if is_gt:
        color=(255, 0, 0)
        thickness = 2
    else:
        color=(0, 0, 255)
        thickness = 2
        
    qs = qs.astype(np.int32)
    # draw 0-1
    cv2.line(image, (qs[0, 0], qs[0, 1]), (qs[1, 0], qs[1, 1]), color, thickness,lineType=cv2.LINE_AA)
    # draw 4-5
    cv2.line(image, (qs[4, 0], qs[4, 1]), (qs[5, 0], qs[5, 1]), color, thickness,lineType=cv2.LINE_AA)
    # draw 0-4
    cv2.line(image, (qs[0, 0], qs[0, 1]), (qs[4, 0], qs[4, 1]), color, thickness,lineType=cv2.LINE_AA)
    # draw 1-5
    cv2.line(image, (qs[1, 0], qs[1, 1]), (qs[5, 0], qs[5, 1]), color, thickness,lineType=cv2.LINE_AA)
    
    # draw 0-3
    cv2.line(image, (qs[0, 0], qs[0, 1]), (qs[3, 0], qs[3, 1]), color, thickness,lineType=cv2.LINE_AA)
    # draw 4-7
    cv2.line(image, (qs[4, 0], qs[4, 1]), (qs[7, 0], qs[7, 1]), color, thickness,lineType=cv2.LINE_AA)
    # draw 1-2
    cv2.line(image, (qs[1, 0], qs[1, 1]), (qs[2, 0], qs[2, 1]), color, thickness,lineType=cv2.LINE_AA)
    # draw 5-6
    cv2.line(image, (qs[5, 0], qs[5, 1]), (qs[6, 0], qs[6, 1]), color, thickness,lineType=cv2.LINE_AA)

    # draw 3-2
    cv2.line(image, (qs[3, 0], qs[3, 1]), (qs[2, 0], qs[2, 1]), color, thickness,lineType=cv2.LINE_AA)
    # draw 6-7
    cv2.line(image, (qs[6, 0], qs[6, 1]), (qs[7, 0], qs[7, 1]), color, thickness,lineType=cv2.LINE_AA)
    # draw 3-7
    cv2.line(image, (qs[3, 0], qs[3, 1]), (qs[7, 0], qs[7, 1]), color, thickness,lineType=cv2.LINE_AA)
    # draw 2-6
    cv2.line(image, (qs[2, 0], qs[2, 1]), (qs[6, 0], qs[6, 1]), color, thickness,lineType=cv2.LINE_AA)
    
    return image
    

def read_img(filename):
    # Convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    return img

def calculate_the_orientation(instances_pointclouds):
    batch_lidar_y_center = []
    batch_oriention = []
    instances_per_bacth = len(instances_pointclouds)
    for i in range(instances_per_bacth):
        y_coor = instances_pointclouds[i][:, 1]
        batch_lidar_y_center.append(y_coor)

        y_thesh = (np.max(y_coor) + np.min(y_coor)) / 2
        y_ind = instances_pointclouds[i][:, 1] > y_thesh

        y_ind_points = instances_pointclouds[i][y_ind]

        if y_ind_points.shape[0] < 10:
            y_ind_points = instances_pointclouds[i]
        

        nums_roi = y_ind_points.shape[0]
        if nums_roi>100:
            rand_ind = np.random.randint(0, y_ind_points.shape[0], 100)
            depth_points_sample = y_ind_points[rand_ind]
        else:
            depth_points_sample = y_ind_points
        # batch_RoI_points[i] = depth_points_sample
        depth_points_np_xz = depth_points_sample[:, [0, 2]]

        '''orient'''
        try:
            orient_set = [(i[1] - j[1]) / (i[0] - j[0]+1e-4) for j in depth_points_np_xz
                            for i in depth_points_np_xz]
            orient_sort = np.array(sorted(np.array(orient_set).reshape(-1)))
            orient_sort = np.arctan(orient_sort[~np.isnan(orient_sort)])
            orient_sort_round = np.around(orient_sort, decimals=1)
            set_orenit = list(set(orient_sort_round))

            if len(set_orenit)==0:
                orient = np.pi/2
            else:
                ind = np.argmax([np.sum(orient_sort_round == i) for i in set_orenit])
                orient = set_orenit[ind]
                if orient < 0:
                    orient += np.pi
                if orient > np.pi / 2 + np.pi * 3 / 8:
                    orient -= np.pi / 2
                if orient < np.pi / 8:
                    orient += np.pi / 2
                    if np.max(instances_pointclouds[i][:, 0]) - np.min(instances_pointclouds[i][:, 0]) > 4 and \
                            (orient >= np.pi / 8 and orient <= np.pi / 2 + np.pi * 3 / 8):
                        if orient < np.pi / 2:
                            orient += np.pi / 2
                        else:
                            orient -= np.pi / 2
        except:
            orient = np.pi
        batch_oriention.append(orient)
    return batch_oriention,batch_lidar_y_center


