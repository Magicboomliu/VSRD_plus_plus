import os
import argparse

import re
import json
import random
import operator
import functools
import shutil
import logging

import torch
import torch.nn as nn
import torchvision
import cv2 as cv
import numpy as np
import scipy as sp

import inflection
import torch.utils.tensorboard

import sys

import vsrd.datasets
sys.path.append("..")

from vsrd import utils
from torch.utils.data import Dataset, DataLoader
from vsrd.datasets.kitti360_dataset import KITTI360Dataset
from vsrd.datasets.transforms import Resizer,MaskAreaFilter,MaskRefiner,BoxGenerator,BoxSizeFilter,SoftRasterizer

# VSRD Dataset
from vsrd.utils import collate_nested_dicts,Dict
# VSRD Networks
from vsrd.models.dynamic_fields.box_residual import BoxParameters3DRBN,ResidualBoxPredictor
from vsrd.models.detectors.box_parameters_with_velocity import BoxParameters3D_With_Velocity
from vsrd.models.detectors.box_parameters_with_scalar_velocity import BoxParameters3D_With_Scalar_Velocity
from vsrd.models.detectors.box_parameters import BoxParameters3D

from vsrd.models.fields import HyperDistanceField
from vsrd.models.encoders import SinusoidalEncoder

# rendering
from vsrd.rendering.utils import ray_casting
from vsrd import rendering

# projection
from vsrd.operations.geometric_operations import project_box_3d
from vsrd.operations.geometric_operations import rotation_matrix,rotation_matrix_x,rotation_matrix_y,rotation_matrix_z
from vsrd.operations.kitti360_operations import box_3d_iou

from vsrd import visualization
from vsrd.utils import Dict
from Optimized_Based.configs import conf_val
import logging
from tqdm import tqdm
import skimage.io

from Optimized_Based.utils.box_geo import decode_box_3d,divide_into_n_parts,get_dynamic_mask_for_the_world_output
from Optimized_Based.utils.file_io import read_text_lines,read_complex_strings
import re
import vsrd

import argparse
import glob
import multiprocessing

import scipy as sp
import pycocotools.mask

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

LINE_INDICES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]

def main(args):

    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "data_2d_raw", "*")))))
    # dynamic_seqences = [sequences[2],sequences[6]] # for ablation studies
    dynamic_seqences = sequences # for all the results


    with multiprocessing.Pool(args.num_workers) as pool:
        with tqdm(total=len(sequences)) as progress_bar:
            for _ in pool.imap_unordered(functools.partial(
                make_predictions,
                root_dirname=args.root_dirname,
                ckpt_dirname=args.ckpt_dirname,
                ckpt_filename=args.ckpt_filename,
                split_dirname=args.split_dirname,
                class_names=args.class_names,
                dynamic_dirname= args.dyanmic_root_filename,
            ), dynamic_seqences):

                progress_bar.update(1)

def make_predictions(
    sequence,
    root_dirname,
    ckpt_dirname,
    ckpt_filename,
    split_dirname,
    dynamic_dirname,
    class_names,):
    
    # group txt
    group_filename = os.path.join(root_dirname, "filenames", split_dirname, sequence, "grouped_image_filenames.txt")
    assert os.path.exists(group_filename)
    with open(group_filename) as file:
        grouped_image_filenames = {
            tuple(map(int, line.split(" ")[0].split(","))): line.split(" ")[1].split(",")
            for line in map(str.strip, file)
        }


    # sample txt
    sample_filename = os.path.join(root_dirname, "filenames", split_dirname, sequence, "sampled_image_filenames.txt")
    assert os.path.exists(sample_filename)
    with open(sample_filename) as file:
        sampled_image_filenames = {
            tuple(map(int, line.split(" ")[0].split(","))): line.split(" ")[1]
            for line in map(str.strip, file)
        }
    
    # dynamic txt
    dynamic_txt_filename = os.path.join(dynamic_dirname,"sync"+sequence[-7:-5],"dynamic_mask.txt")
    assert os.path.exists(dynamic_txt_filename)
    with open(dynamic_txt_filename) as file:
        dynamic_instance_list = {
            tuple(map(int, line.split(" ")[0].split(","))): [int(float(item)) for item in line.split(" ")[2].split(",")] 
            for line in map(str.strip, file)
        }
        
    
    for instance_ids, grouped_image_filenames in grouped_image_filenames.items():
        
        # get the target image filename
        target_image_filename = sampled_image_filenames[instance_ids]
        # get the instance dynamic list
        instance_dynamic_list = dynamic_instance_list[instance_ids]
        
        # image direction filenames
        target_image_dirname = os.path.splitext(os.path.relpath(target_image_filename, root_dirname))[0]
        # get the models ckpts
        target_ckpt_filename = os.path.join(ckpt_dirname, target_image_dirname, ckpt_filename)


        if not os.path.exists(target_ckpt_filename):
            print(f"[{target_ckpt_filename}] Does not exist!")
            continue

        assert os.path.exists(target_ckpt_filename)
        assert os.path.exists(target_image_filename)

        target_checkpoint = torch.load(target_ckpt_filename, map_location="cpu")
        num_instances = target_checkpoint["models"]["detector"]["embeddings"].shape[1]
        models_weights_new = target_checkpoint['models']
        
        # load the pretrained models
        if args.input_model_type=='velocity_with_init':
            
            models = Dict()
            detector = BoxParameters3D_With_Velocity(batch_size=1,
                                num_instances=num_instances,
                                num_features=256)
            hyper_distance_field = HyperDistanceField(in_channels=48,
                                                    out_channels_list=[16,16,16,16],
                                                    hyper_in_channels=256,
                                                    hyper_out_channels_list=[256,256,256,256])
            positional_encoder = SinusoidalEncoder(num_frequencies=8)
            models['detector'] = detector
            models['hyper_distance_field'] = hyper_distance_field
            models['positional_encoder'] = positional_encoder
            for model in models.values():
                model.to(0)
                
            # Loaded the Weights from the Per-trained Modelss
            for model_name, model in models.items():
                models[model_name].load_state_dict(models_weights_new[model_name])
            # print("Loaded all the pre-trained Models for Name {}.".format(target_ckpt_filename))
            '''---------------------------------------------------------------------------------------'''
            
            '''Loading the Model is Finished....'''
            world_outputs = utils.Dict.apply(models.detector())  # dict_keys(['boxes_3d', 'locations', 'dimensions', 'orientations', 'embeddings'])
            multi_outputs = utils.DefaultDict(utils.Dict)
            world_boxes_3d = nn.functional.pad(world_outputs.boxes_3d, (0, 1), mode="constant", value=1.0) #(1,4,8,4)
            world_boxes_3d = world_boxes_3d.cpu()

            current_velocity = models['detector'].velocity #[1,8,3]
            current_velocity = current_velocity.unsqueeze(-2).cpu() #[1,1,1,3]

        
            # Loaded the target annotation filenames path
            target_annotation_filename = target_image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")
            assert os.path.exists(target_annotation_filename)
            
            with open(target_annotation_filename) as file:
                target_annotation = json.load(file)
            
            target_readed_instance_ids = {
                class_name: list(masks.keys())
                for class_name, masks in target_annotation["masks"].items()
                if class_name in class_names
                    }['car']
            target_readed_instance_ids = tuple([int(float(item)) for item in target_readed_instance_ids])



            target_extrinsic_matrix = torch.tensor(target_annotation["extrinsic_matrix"])
            inverse_target_extrinsic_matrix = torch.linalg.inv(target_extrinsic_matrix) #[4,4]

            x_axis, y_axis, _ = target_extrinsic_matrix[..., :3, :3]
            rectification_angle = (
                torch.acos(torch.dot(torch.round(y_axis), y_axis)) *
                torch.sign(torch.dot(torch.cross(torch.round(y_axis), y_axis), x_axis))
            )
            rectification_matrix = vsrd.operations.rotation_matrix_x(rectification_angle)
            target_instance_ids = torch.cat([
                torch.as_tensor(list(map(int, masks.keys())), dtype=torch.long)
                for class_name, masks in target_annotation["masks"].items()
                if class_name in class_names
            ], dim=0)
            
            
            accumulated_iou_matrix = torch.zeros(len(world_boxes_3d.squeeze(0)), len(target_instance_ids))
            accumulated_cnt_matrix = torch.zeros(len(world_boxes_3d.squeeze(0)), len(target_instance_ids))


            # target image extrinsic and intrinsic
            mine_current_extrinsic_matrices =(
                target_extrinsic_matrix@
                inverse_target_extrinsic_matrix @
                vsrd.operations.expand_to_4x4(rectification_matrix.T))
            mine_current_intrinsic_matrices = torch.tensor(target_annotation["intrinsic_matrix"])
            mine_gt_masks = torch.cat([
                torch.as_tensor(np.stack(list(map(pycocotools.mask.decode, masks.values()))), dtype=torch.float)
                for class_name, masks in target_annotation["masks"].items()
                if class_name in class_names], dim=0)

            mine_gt_masks = vsrd.transforms.MaskRefiner()(mine_gt_masks)["masks"]
            mine_gt_boxes_2d = torchvision.ops.masks_to_boxes(mine_gt_masks.bool()).unflatten(-1, (2, 2))
            
            # get the instance ids
            
            if len(target_readed_instance_ids)!=len(instance_ids):
                index = [target_readed_instance_ids.index(ids) for ids in instance_ids]
                mine_gt_boxes_2d = mine_gt_boxes_2d[index]
  
                

            # Re-Ordered the Dynamic List
            instance_dynamic_list = Re_Ordered_the_Dyanmic(target_input_extrinsic_matrices=mine_current_extrinsic_matrices,
                                   target_input_intrinsic_matrices=mine_current_intrinsic_matrices,
                                   target_inputs_boxes_2d=mine_gt_boxes_2d,
                                   world_boxes_3d=world_boxes_3d,
                                   original_dynamic_list=instance_dynamic_list)
            
            
            callbacks = []

            for source_image_filename in grouped_image_filenames:

                source_annotation_filename = source_image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")
                assert os.path.exists(source_annotation_filename)
                
                with open(source_annotation_filename) as file:
                    source_annotation = json.load(file)
                
                # source intrinsic matrix
                source_intrinsic_matrix = torch.tensor(source_annotation["intrinsic_matrix"])
                source_extrinsic_matrix = torch.tensor(source_annotation["extrinsic_matrix"])

                # change to relative pose.
                source_extrinsic_matrix = (
                    source_extrinsic_matrix @
                    inverse_target_extrinsic_matrix @
                    vsrd.operations.expand_to_4x4(rectification_matrix.T)
                )
                
                # get the relative id
                source_image_fname = int(os.path.basename(source_image_filename)[:-4])
                target_image_fname = int(os.path.basename(target_image_filename)[:-4])
                
                relative_frame_timestep = source_image_fname - target_image_fname
                relative_box_residual = current_velocity * relative_frame_timestep 
                
                instance_dynamic_for_output_tensor = torch.from_numpy(np.array(instance_dynamic_list)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(relative_box_residual.device).float()
                relative_box_residual = relative_box_residual * instance_dynamic_for_output_tensor
                relative_box_residual = relative_box_residual.squeeze(-2)
                
                

                # bounding boxes withj alignment
                current_world_boxes_3d = decode_box_3d(locations=world_outputs.locations.cpu(),
                                dimensions=world_outputs.dimensions.cpu(),
                                orientations=world_outputs.orientations.cpu(),
                                residual= relative_box_residual)
                

                current_world_boxes_3d = nn.functional.pad(current_world_boxes_3d, (0, 1), mode="constant", value=1.0) #(1,4,8,4)
                

                source_pd_boxes_3d = current_world_boxes_3d @ source_extrinsic_matrix.T
                source_pd_boxes_3d = source_pd_boxes_3d[..., :-1] / source_pd_boxes_3d[..., -1:]
                
                
                source_pd_boxes_3d = source_pd_boxes_3d.squeeze(0)
                
                source_pd_boxes_2d = torch.stack([
                    vsrd.operations.project_box_3d(
                        box_3d=source_pd_box_3d,
                        line_indices=LINE_INDICES,
                        intrinsic_matrix=source_intrinsic_matrix,
                    )
                    for source_pd_box_3d in source_pd_boxes_3d
                ], dim=0)
                
                source_gt_masks = torch.cat([
                    torch.as_tensor(np.stack(list(map(pycocotools.mask.decode, masks.values()))), dtype=torch.float)
                    for class_name, masks in source_annotation["masks"].items()
                    if class_name in class_names
                ], dim=0)

                source_gt_masks = vsrd.transforms.MaskRefiner()(source_gt_masks)["masks"]
                source_gt_boxes_2d = torchvision.ops.masks_to_boxes(source_gt_masks.bool()).unflatten(-1, (2, 2))


                source_pd_boxes_2d = torchvision.ops.clip_boxes_to_image(
                    boxes=source_pd_boxes_2d.flatten(-2, -1),
                    size=source_gt_masks.shape[-2:],
                ).unflatten(-1, (2, 2))
                
                


                source_iou_matrix = torch.nan_to_num(torchvision.ops.box_iou(
                    boxes1=source_pd_boxes_2d.flatten(-2, -1),
                    boxes2=source_gt_boxes_2d.flatten(-2, -1),
                ))

                source_instance_ids = torch.cat([
                    torch.as_tensor(list(map(int, masks.keys())), dtype=torch.long)
                    for class_name, masks in source_annotation["masks"].items()
                    if class_name in class_names
                ], dim=0)

                target_gt_indices = source_instance_ids.new_tensor([
                    target_instance_ids.tolist().index(source_instance_id.item())
                    if source_instance_id in target_instance_ids else -1
                    for source_instance_id in source_instance_ids
                ])

                accumulated_iou_matrix[
                    ...,
                    target_gt_indices[target_gt_indices >= 0]
                ] += source_iou_matrix[..., target_gt_indices >= 0]

                accumulated_cnt_matrix[
                    ...,
                    target_gt_indices[target_gt_indices >= 0]
                ] += 1
                
                def save_prediction(filename, boxes_3d, boxes_2d, confidences):

                    prediction = dict(
                        boxes_3d=dict(car=boxes_3d.tolist()),
                        boxes_2d=dict(car=boxes_2d.tolist()),
                        confidences=dict(car=confidences.tolist()),
                    )

                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, "w") as file:
                        json.dump(prediction, file, indent=4, sort_keys=False)

                source_prediction_dirname = os.path.join(args.saved_pseudo_folder_path, os.path.basename(ckpt_dirname))
                source_prediction_filename = source_annotation_filename.replace("annotations", source_prediction_dirname)
                

                callbacks.append(functools.partial(
                    save_prediction,
                    filename=source_prediction_filename,
                    boxes_3d=source_pd_boxes_3d,
                    boxes_2d=source_pd_boxes_2d,
                ))

            averaged_iou_matrix = accumulated_iou_matrix / accumulated_cnt_matrix
            matched_pd_indices, matched_gt_indices = vsrd.utils.torch_function(sp.optimize.linear_sum_assignment)(averaged_iou_matrix, maximize=True)

            confidences = averaged_iou_matrix[matched_pd_indices, matched_gt_indices]

            for callback in callbacks:
                callback(confidences=confidences)
        
        # load the pretrained models
        if args.input_model_type =="vanilla":
            models = Dict()
            detector = BoxParameters3D(batch_size=1,
                                       num_instances=num_instances,
                                       num_features=256)
            hyper_distance_field = HyperDistanceField(in_channels=48,
                                                    out_channels_list=[16,16,16,16],
                                                    hyper_in_channels=256,
                                                    hyper_out_channels_list=[256,256,256,256])
            positional_encoder = SinusoidalEncoder(num_frequencies=8)
            models['detector'] = detector
            models['hyper_distance_field'] = hyper_distance_field
            models['positional_encoder'] = positional_encoder
            for model in models.values():
                model.to(0)
                
            # Loaded the Weights from the Per-trained Modelss
            for model_name, model in models.items():
                models[model_name].load_state_dict(models_weights_new[model_name])
            # print("Loaded all the pre-trained Models for Name {}.".format(target_ckpt_filename))
            '''---------------------------------------------------------------------------------------'''

            '''Loading the Model is Finished....'''
            world_outputs = utils.Dict.apply(models.detector())  # dict_keys(['boxes_3d', 'locations', 'dimensions', 'orientations', 'embeddings'])
            multi_outputs = utils.DefaultDict(utils.Dict)
            world_boxes_3d = nn.functional.pad(world_outputs.boxes_3d, (0, 1), mode="constant", value=1.0) #(1,4,8,4)
            world_boxes_3d = world_boxes_3d.cpu()


            # Loaded the target annotation filenames path
            target_annotation_filename = target_image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")
            assert os.path.exists(target_annotation_filename)
            
            with open(target_annotation_filename) as file:
                target_annotation = json.load(file)
            
            target_readed_instance_ids = {
                class_name: list(masks.keys())
                for class_name, masks in target_annotation["masks"].items()
                if class_name in class_names
                    }['car']
            target_readed_instance_ids = tuple([int(float(item)) for item in target_readed_instance_ids])



            target_extrinsic_matrix = torch.tensor(target_annotation["extrinsic_matrix"])
            inverse_target_extrinsic_matrix = torch.linalg.inv(target_extrinsic_matrix) #[4,4]

            x_axis, y_axis, _ = target_extrinsic_matrix[..., :3, :3]
            rectification_angle = (
                torch.acos(torch.dot(torch.round(y_axis), y_axis)) *
                torch.sign(torch.dot(torch.cross(torch.round(y_axis), y_axis), x_axis))
            )
            rectification_matrix = vsrd.operations.rotation_matrix_x(rectification_angle)
            target_instance_ids = torch.cat([
                torch.as_tensor(list(map(int, masks.keys())), dtype=torch.long)
                for class_name, masks in target_annotation["masks"].items()
                if class_name in class_names
            ], dim=0)
            
            
            accumulated_iou_matrix = torch.zeros(len(world_boxes_3d.squeeze(0)), len(target_instance_ids))
            accumulated_cnt_matrix = torch.zeros(len(world_boxes_3d.squeeze(0)), len(target_instance_ids))


            # target image extrinsic and intrinsic
            mine_current_extrinsic_matrices =(
                target_extrinsic_matrix@
                inverse_target_extrinsic_matrix @
                vsrd.operations.expand_to_4x4(rectification_matrix.T))
            mine_current_intrinsic_matrices = torch.tensor(target_annotation["intrinsic_matrix"])
            mine_gt_masks = torch.cat([
                torch.as_tensor(np.stack(list(map(pycocotools.mask.decode, masks.values()))), dtype=torch.float)
                for class_name, masks in target_annotation["masks"].items()
                if class_name in class_names], dim=0)

            mine_gt_masks = vsrd.transforms.MaskRefiner()(mine_gt_masks)["masks"]
            mine_gt_boxes_2d = torchvision.ops.masks_to_boxes(mine_gt_masks.bool()).unflatten(-1, (2, 2))
            
            # get the instance ids
            
            if len(target_readed_instance_ids)!=len(instance_ids):
                index = [target_readed_instance_ids.index(ids) for ids in instance_ids]
                mine_gt_boxes_2d = mine_gt_boxes_2d[index]

            
            
            callbacks = []

            for source_image_filename in grouped_image_filenames:

                source_annotation_filename = source_image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")
                assert os.path.exists(source_annotation_filename)
                
                with open(source_annotation_filename) as file:
                    source_annotation = json.load(file)
                
                # source intrinsic matrix
                source_intrinsic_matrix = torch.tensor(source_annotation["intrinsic_matrix"])
                source_extrinsic_matrix = torch.tensor(source_annotation["extrinsic_matrix"])

                # change to relative pose.
                source_extrinsic_matrix = (
                    source_extrinsic_matrix @
                    inverse_target_extrinsic_matrix @
                    vsrd.operations.expand_to_4x4(rectification_matrix.T)
                )
                
                # get the relative id
                source_image_fname = int(os.path.basename(source_image_filename)[:-4])
                target_image_fname = int(os.path.basename(target_image_filename)[:-4])
                
                

                # bounding boxes withj alignment
                current_world_boxes_3d = world_boxes_3d

                source_pd_boxes_3d = current_world_boxes_3d @ source_extrinsic_matrix.T
                source_pd_boxes_3d = source_pd_boxes_3d[..., :-1] / source_pd_boxes_3d[..., -1:]
                source_pd_boxes_3d = source_pd_boxes_3d.squeeze(0)
                
                source_pd_boxes_2d = torch.stack([
                    vsrd.operations.project_box_3d(
                        box_3d=source_pd_box_3d,
                        line_indices=LINE_INDICES,
                        intrinsic_matrix=source_intrinsic_matrix,
                    )
                    for source_pd_box_3d in source_pd_boxes_3d
                ], dim=0)
                
                

                source_gt_masks = torch.cat([
                    torch.as_tensor(np.stack(list(map(pycocotools.mask.decode, masks.values()))), dtype=torch.float)
                    for class_name, masks in source_annotation["masks"].items()
                    if class_name in class_names
                ], dim=0)

                source_gt_masks = vsrd.transforms.MaskRefiner()(source_gt_masks)["masks"]
                source_gt_boxes_2d = torchvision.ops.masks_to_boxes(source_gt_masks.bool()).unflatten(-1, (2, 2))


                source_pd_boxes_2d = torchvision.ops.clip_boxes_to_image(
                    boxes=source_pd_boxes_2d.flatten(-2, -1),
                    size=source_gt_masks.shape[-2:],
                ).unflatten(-1, (2, 2))
                
                


                source_iou_matrix = torch.nan_to_num(torchvision.ops.box_iou(
                    boxes1=source_pd_boxes_2d.flatten(-2, -1),
                    boxes2=source_gt_boxes_2d.flatten(-2, -1),
                ))

                source_instance_ids = torch.cat([
                    torch.as_tensor(list(map(int, masks.keys())), dtype=torch.long)
                    for class_name, masks in source_annotation["masks"].items()
                    if class_name in class_names
                ], dim=0)

                target_gt_indices = source_instance_ids.new_tensor([
                    target_instance_ids.tolist().index(source_instance_id.item())
                    if source_instance_id in target_instance_ids else -1
                    for source_instance_id in source_instance_ids
                ])

                accumulated_iou_matrix[
                    ...,
                    target_gt_indices[target_gt_indices >= 0]
                ] += source_iou_matrix[..., target_gt_indices >= 0]

                accumulated_cnt_matrix[
                    ...,
                    target_gt_indices[target_gt_indices >= 0]
                ] += 1
                
                def save_prediction(filename, boxes_3d, boxes_2d, confidences):

                    prediction = dict(
                        boxes_3d=dict(car=boxes_3d.tolist()),
                        boxes_2d=dict(car=boxes_2d.tolist()),
                        confidences=dict(car=confidences.tolist()),
                    )

                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, "w") as file:
                        json.dump(prediction, file, indent=4, sort_keys=False)

                source_prediction_dirname = os.path.join(args.saved_pseudo_folder_path, os.path.basename(ckpt_dirname))
                source_prediction_filename = source_annotation_filename.replace("annotations", source_prediction_dirname)
                

                callbacks.append(functools.partial(
                    save_prediction,
                    filename=source_prediction_filename,
                    boxes_3d=source_pd_boxes_3d,
                    boxes_2d=source_pd_boxes_2d,
                ))

            averaged_iou_matrix = accumulated_iou_matrix / accumulated_cnt_matrix
            matched_pd_indices, matched_gt_indices = vsrd.utils.torch_function(sp.optimize.linear_sum_assignment)(averaged_iou_matrix, maximize=True)

            confidences = averaged_iou_matrix[matched_pd_indices, matched_gt_indices]

            for callback in callbacks:
                callback(confidences=confidences)
        
        # load the pre-trained models
        if args.input_model_type =="velocity_only":
            
            models = Dict()
            detector = BoxParameters3D_With_Velocity(batch_size=1,
                                num_instances=num_instances,
                                num_features=256)
            hyper_distance_field = HyperDistanceField(in_channels=48,
                                                    out_channels_list=[16,16,16,16],
                                                    hyper_in_channels=256,
                                                    hyper_out_channels_list=[256,256,256,256])
            positional_encoder = SinusoidalEncoder(num_frequencies=8)
            models['detector'] = detector
            models['hyper_distance_field'] = hyper_distance_field
            models['positional_encoder'] = positional_encoder
            for model in models.values():
                model.to(0)
                
            # Loaded the Weights from the Per-trained Modelss
            for model_name, model in models.items():
                models[model_name].load_state_dict(models_weights_new[model_name])
            # print("Loaded all the pre-trained Models for Name {}.".format(target_ckpt_filename))
            '''---------------------------------------------------------------------------------------'''
            
            '''Loading the Model is Finished....'''
            world_outputs = utils.Dict.apply(models.detector())  # dict_keys(['boxes_3d', 'locations', 'dimensions', 'orientations', 'embeddings'])
            multi_outputs = utils.DefaultDict(utils.Dict)
            world_boxes_3d = nn.functional.pad(world_outputs.boxes_3d, (0, 1), mode="constant", value=1.0) #(1,4,8,4)
            world_boxes_3d = world_boxes_3d.cpu()

            current_velocity = models['detector'].velocity #[1,8,3]
            current_velocity = current_velocity.unsqueeze(-2).cpu() #[1,1,1,3]

        
            # Loaded the target annotation filenames path
            target_annotation_filename = target_image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")
            assert os.path.exists(target_annotation_filename)
            
            with open(target_annotation_filename) as file:
                target_annotation = json.load(file)
            
            target_readed_instance_ids = {
                class_name: list(masks.keys())
                for class_name, masks in target_annotation["masks"].items()
                if class_name in class_names
                    }['car']
            target_readed_instance_ids = tuple([int(float(item)) for item in target_readed_instance_ids])



            target_extrinsic_matrix = torch.tensor(target_annotation["extrinsic_matrix"])
            inverse_target_extrinsic_matrix = torch.linalg.inv(target_extrinsic_matrix) #[4,4]

            x_axis, y_axis, _ = target_extrinsic_matrix[..., :3, :3]
            rectification_angle = (
                torch.acos(torch.dot(torch.round(y_axis), y_axis)) *
                torch.sign(torch.dot(torch.cross(torch.round(y_axis), y_axis), x_axis))
            )
            rectification_matrix = vsrd.operations.rotation_matrix_x(rectification_angle)
            target_instance_ids = torch.cat([
                torch.as_tensor(list(map(int, masks.keys())), dtype=torch.long)
                for class_name, masks in target_annotation["masks"].items()
                if class_name in class_names
            ], dim=0)
            
            
            accumulated_iou_matrix = torch.zeros(len(world_boxes_3d.squeeze(0)), len(target_instance_ids))
            accumulated_cnt_matrix = torch.zeros(len(world_boxes_3d.squeeze(0)), len(target_instance_ids))


            # target image extrinsic and intrinsic
            mine_current_extrinsic_matrices =(
                target_extrinsic_matrix@
                inverse_target_extrinsic_matrix @
                vsrd.operations.expand_to_4x4(rectification_matrix.T))
            mine_current_intrinsic_matrices = torch.tensor(target_annotation["intrinsic_matrix"])
            mine_gt_masks = torch.cat([
                torch.as_tensor(np.stack(list(map(pycocotools.mask.decode, masks.values()))), dtype=torch.float)
                for class_name, masks in target_annotation["masks"].items()
                if class_name in class_names], dim=0)

            mine_gt_masks = vsrd.transforms.MaskRefiner()(mine_gt_masks)["masks"]
            mine_gt_boxes_2d = torchvision.ops.masks_to_boxes(mine_gt_masks.bool()).unflatten(-1, (2, 2))
            
            # get the instance ids
            
            if len(target_readed_instance_ids)!=len(instance_ids):
                index = [target_readed_instance_ids.index(ids) for ids in instance_ids]
                mine_gt_boxes_2d = mine_gt_boxes_2d[index]
  
                

            # Re-Ordered the Dynamic List
            instance_dynamic_list = Re_Ordered_the_Dyanmic(target_input_extrinsic_matrices=mine_current_extrinsic_matrices,
                                   target_input_intrinsic_matrices=mine_current_intrinsic_matrices,
                                   target_inputs_boxes_2d=mine_gt_boxes_2d,
                                   world_boxes_3d=world_boxes_3d,
                                   original_dynamic_list=instance_dynamic_list)
            
            
            callbacks = []

            for source_image_filename in grouped_image_filenames:

                source_annotation_filename = source_image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")
                assert os.path.exists(source_annotation_filename)
                
                with open(source_annotation_filename) as file:
                    source_annotation = json.load(file)
                
                # source intrinsic matrix
                source_intrinsic_matrix = torch.tensor(source_annotation["intrinsic_matrix"])
                source_extrinsic_matrix = torch.tensor(source_annotation["extrinsic_matrix"])

                # change to relative pose.
                source_extrinsic_matrix = (
                    source_extrinsic_matrix @
                    inverse_target_extrinsic_matrix @
                    vsrd.operations.expand_to_4x4(rectification_matrix.T)
                )
                
                # get the relative id
                source_image_fname = int(os.path.basename(source_image_filename)[:-4])
                target_image_fname = int(os.path.basename(target_image_filename)[:-4])
                
                relative_frame_timestep = source_image_fname - target_image_fname
                relative_box_residual = current_velocity * relative_frame_timestep 
                
                instance_dynamic_for_output_tensor = torch.from_numpy(np.array(instance_dynamic_list)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(relative_box_residual.device).float()
                relative_box_residual = relative_box_residual * instance_dynamic_for_output_tensor
                relative_box_residual = relative_box_residual.squeeze(-2)
                
                

                # bounding boxes withj alignment
                current_world_boxes_3d = decode_box_3d(locations=world_outputs.locations.cpu(),
                                dimensions=world_outputs.dimensions.cpu(),
                                orientations=world_outputs.orientations.cpu(),
                                residual= relative_box_residual)
                

                current_world_boxes_3d = nn.functional.pad(current_world_boxes_3d, (0, 1), mode="constant", value=1.0) #(1,4,8,4)
                

                source_pd_boxes_3d = current_world_boxes_3d @ source_extrinsic_matrix.T
                source_pd_boxes_3d = source_pd_boxes_3d[..., :-1] / source_pd_boxes_3d[..., -1:]
                
                
                source_pd_boxes_3d = source_pd_boxes_3d.squeeze(0)
                
                source_pd_boxes_2d = torch.stack([
                    vsrd.operations.project_box_3d(
                        box_3d=source_pd_box_3d,
                        line_indices=LINE_INDICES,
                        intrinsic_matrix=source_intrinsic_matrix,
                    )
                    for source_pd_box_3d in source_pd_boxes_3d
                ], dim=0)
                
                

                source_gt_masks = torch.cat([
                    torch.as_tensor(np.stack(list(map(pycocotools.mask.decode, masks.values()))), dtype=torch.float)
                    for class_name, masks in source_annotation["masks"].items()
                    if class_name in class_names
                ], dim=0)

                source_gt_masks = vsrd.transforms.MaskRefiner()(source_gt_masks)["masks"]
                source_gt_boxes_2d = torchvision.ops.masks_to_boxes(source_gt_masks.bool()).unflatten(-1, (2, 2))


                source_pd_boxes_2d = torchvision.ops.clip_boxes_to_image(
                    boxes=source_pd_boxes_2d.flatten(-2, -1),
                    size=source_gt_masks.shape[-2:],
                ).unflatten(-1, (2, 2))
                
                


                source_iou_matrix = torch.nan_to_num(torchvision.ops.box_iou(
                    boxes1=source_pd_boxes_2d.flatten(-2, -1),
                    boxes2=source_gt_boxes_2d.flatten(-2, -1),
                ))

                source_instance_ids = torch.cat([
                    torch.as_tensor(list(map(int, masks.keys())), dtype=torch.long)
                    for class_name, masks in source_annotation["masks"].items()
                    if class_name in class_names
                ], dim=0)

                target_gt_indices = source_instance_ids.new_tensor([
                    target_instance_ids.tolist().index(source_instance_id.item())
                    if source_instance_id in target_instance_ids else -1
                    for source_instance_id in source_instance_ids
                ])

                accumulated_iou_matrix[
                    ...,
                    target_gt_indices[target_gt_indices >= 0]
                ] += source_iou_matrix[..., target_gt_indices >= 0]

                accumulated_cnt_matrix[
                    ...,
                    target_gt_indices[target_gt_indices >= 0]
                ] += 1
                
                def save_prediction(filename, boxes_3d, boxes_2d, confidences):

                    prediction = dict(
                        boxes_3d=dict(car=boxes_3d.tolist()),
                        boxes_2d=dict(car=boxes_2d.tolist()),
                        confidences=dict(car=confidences.tolist()),
                    )

                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, "w") as file:
                        json.dump(prediction, file, indent=4, sort_keys=False)

                source_prediction_dirname = os.path.join(args.saved_pseudo_folder_path, os.path.basename(ckpt_dirname))
                source_prediction_filename = source_annotation_filename.replace("annotations", source_prediction_dirname)
                

                callbacks.append(functools.partial(
                    save_prediction,
                    filename=source_prediction_filename,
                    boxes_3d=source_pd_boxes_3d,
                    boxes_2d=source_pd_boxes_2d,
                ))

            averaged_iou_matrix = accumulated_iou_matrix / accumulated_cnt_matrix
            matched_pd_indices, matched_gt_indices = vsrd.utils.torch_function(sp.optimize.linear_sum_assignment)(averaged_iou_matrix, maximize=True)

            confidences = averaged_iou_matrix[matched_pd_indices, matched_gt_indices]

            for callback in callbacks:
                callback(confidences=confidences)
        
        # load the pre-trained models
        # load the pre-trained models
        if args.input_model_type =="mlp_only":
            
            models = Dict()
            detector = BoxParameters3DRBN(batch_size=1,
                                       num_instances=num_instances,
                                       num_features=256)
            
            box_residual_detector = ResidualBoxPredictor()
            hyper_distance_field = HyperDistanceField(in_channels=48,
                                                    out_channels_list=[16,16,16,16],
                                                    hyper_in_channels=256,
                                                    hyper_out_channels_list=[256,256,256,256])
            positional_encoder = SinusoidalEncoder(num_frequencies=8)
            models['detector'] = detector
            models['hyper_distance_field'] = hyper_distance_field
            models['positional_encoder'] = positional_encoder
            models['detector_residual'] = box_residual_detector
            for model in models.values():
                model.to(0)
                model.eval()
                
            # Loaded the Weights from the Per-trained Modelss
            for model_name, model in models.items():
                models[model_name].load_state_dict(models_weights_new[model_name])
            # print("Loaded all the pre-trained Models for Name {}.".format(target_ckpt_filename))
            '''---------------------------------------------------------------------------------------'''
            
            '''Loading the Model is Finished....'''
            world_outputs = utils.Dict.apply(models.detector())  # dict_keys(['boxes_3d', 'locations', 'dimensions', 'orientations', 'embeddings'])
            multi_outputs = utils.DefaultDict(utils.Dict)
            world_boxes_3d = nn.functional.pad(world_outputs.boxes_3d, (0, 1), mode="constant", value=1.0) #(1,4,8,4)
            world_boxes_3d = world_boxes_3d.cpu()

        
            # Loaded the target annotation filenames path
            target_annotation_filename = target_image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")
            assert os.path.exists(target_annotation_filename)
            with open(target_annotation_filename) as file:
                target_annotation = json.load(file)
            
            target_readed_instance_ids = {
                class_name: list(masks.keys())
                for class_name, masks in target_annotation["masks"].items()
                if class_name in class_names
                    }['car']
            target_readed_instance_ids = tuple([int(float(item)) for item in target_readed_instance_ids])
            target_extrinsic_matrix = torch.tensor(target_annotation["extrinsic_matrix"])
            inverse_target_extrinsic_matrix = torch.linalg.inv(target_extrinsic_matrix) #[4,4]

            x_axis, y_axis, _ = target_extrinsic_matrix[..., :3, :3]
            rectification_angle = (
                torch.acos(torch.dot(torch.round(y_axis), y_axis)) *
                torch.sign(torch.dot(torch.cross(torch.round(y_axis), y_axis), x_axis))
            )
            rectification_matrix = vsrd.operations.rotation_matrix_x(rectification_angle)
            target_instance_ids = torch.cat([
                torch.as_tensor(list(map(int, masks.keys())), dtype=torch.long)
                for class_name, masks in target_annotation["masks"].items()
                if class_name in class_names
            ], dim=0)
            
            
            accumulated_iou_matrix = torch.zeros(len(world_boxes_3d.squeeze(0)), len(target_instance_ids))
            accumulated_cnt_matrix = torch.zeros(len(world_boxes_3d.squeeze(0)), len(target_instance_ids))


            # target image extrinsic and intrinsic
            mine_current_extrinsic_matrices =(
                target_extrinsic_matrix@
                inverse_target_extrinsic_matrix @
                vsrd.operations.expand_to_4x4(rectification_matrix.T))
            mine_current_intrinsic_matrices = torch.tensor(target_annotation["intrinsic_matrix"])
            mine_gt_masks = torch.cat([
                torch.as_tensor(np.stack(list(map(pycocotools.mask.decode, masks.values()))), dtype=torch.float)
                for class_name, masks in target_annotation["masks"].items()
                if class_name in class_names], dim=0)

            mine_gt_masks = vsrd.transforms.MaskRefiner()(mine_gt_masks)["masks"]
            mine_gt_boxes_2d = torchvision.ops.masks_to_boxes(mine_gt_masks.bool()).unflatten(-1, (2, 2))
            
            # get the instance ids
            
            if len(target_readed_instance_ids)!=len(instance_ids):
                index = [target_readed_instance_ids.index(ids) for ids in instance_ids]
                mine_gt_boxes_2d = mine_gt_boxes_2d[index]
  
                

            
            callbacks = []

            for source_image_filename in grouped_image_filenames:

                source_annotation_filename = source_image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")
                assert os.path.exists(source_annotation_filename)
                
                with open(source_annotation_filename) as file:
                    source_annotation = json.load(file)
                
                # source intrinsic matrix
                source_intrinsic_matrix = torch.tensor(source_annotation["intrinsic_matrix"])
                source_extrinsic_matrix = torch.tensor(source_annotation["extrinsic_matrix"])

                # change to relative pose.
                source_extrinsic_matrix = (
                    source_extrinsic_matrix @
                    inverse_target_extrinsic_matrix @
                    vsrd.operations.expand_to_4x4(rectification_matrix.T)
                )
                
                # get the relative id
                source_image_fname = int(os.path.basename(source_image_filename)[:-4])
                target_image_fname = int(os.path.basename(target_image_filename)[:-4])
                
                relative_frame_timestep = source_image_fname - target_image_fname

        
                relative_box_residual = models.detector_residual([relative_frame_timestep],world_outputs.embeddings)
                relative_box_residual = relative_box_residual.squeeze(-2).cpu()
      

                # bounding boxes withj alignment
                current_world_boxes_3d = decode_box_3d(locations=world_outputs.locations.cpu(),
                                dimensions=world_outputs.dimensions.cpu(),
                                orientations=world_outputs.orientations.cpu(),
                                residual= relative_box_residual)
                

                current_world_boxes_3d = nn.functional.pad(current_world_boxes_3d, (0, 1), mode="constant", value=1.0) #(1,4,8,4)
                

                source_pd_boxes_3d = current_world_boxes_3d @ source_extrinsic_matrix.T
                source_pd_boxes_3d = source_pd_boxes_3d[..., :-1] / source_pd_boxes_3d[..., -1:]
                
                
                source_pd_boxes_3d = source_pd_boxes_3d.squeeze(0)
                
                source_pd_boxes_2d = torch.stack([
                    vsrd.operations.project_box_3d(
                        box_3d=source_pd_box_3d,
                        line_indices=LINE_INDICES,
                        intrinsic_matrix=source_intrinsic_matrix,
                    )
                    for source_pd_box_3d in source_pd_boxes_3d
                ], dim=0)
                
                

                source_gt_masks = torch.cat([
                    torch.as_tensor(np.stack(list(map(pycocotools.mask.decode, masks.values()))), dtype=torch.float)
                    for class_name, masks in source_annotation["masks"].items()
                    if class_name in class_names
                ], dim=0)

                source_gt_masks = vsrd.transforms.MaskRefiner()(source_gt_masks)["masks"]
                source_gt_boxes_2d = torchvision.ops.masks_to_boxes(source_gt_masks.bool()).unflatten(-1, (2, 2))


                source_pd_boxes_2d = torchvision.ops.clip_boxes_to_image(
                    boxes=source_pd_boxes_2d.flatten(-2, -1),
                    size=source_gt_masks.shape[-2:],
                ).unflatten(-1, (2, 2))
                
                


                source_iou_matrix = torch.nan_to_num(torchvision.ops.box_iou(
                    boxes1=source_pd_boxes_2d.flatten(-2, -1),
                    boxes2=source_gt_boxes_2d.flatten(-2, -1),
                ))

                source_instance_ids = torch.cat([
                    torch.as_tensor(list(map(int, masks.keys())), dtype=torch.long)
                    for class_name, masks in source_annotation["masks"].items()
                    if class_name in class_names
                ], dim=0)

                target_gt_indices = source_instance_ids.new_tensor([
                    target_instance_ids.tolist().index(source_instance_id.item())
                    if source_instance_id in target_instance_ids else -1
                    for source_instance_id in source_instance_ids
                ])

                accumulated_iou_matrix[
                    ...,
                    target_gt_indices[target_gt_indices >= 0]
                ] += source_iou_matrix[..., target_gt_indices >= 0]

                accumulated_cnt_matrix[
                    ...,
                    target_gt_indices[target_gt_indices >= 0]
                ] += 1
                
                def save_prediction(filename, boxes_3d, boxes_2d, confidences):

                    prediction = dict(
                        boxes_3d=dict(car=boxes_3d.tolist()),
                        boxes_2d=dict(car=boxes_2d.tolist()),
                        confidences=dict(car=confidences.tolist()),
                    )

                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, "w") as file:
                        json.dump(prediction, file, indent=4, sort_keys=False)

                source_prediction_dirname = os.path.join(args.saved_pseudo_folder_path, os.path.basename(ckpt_dirname))
                source_prediction_filename = source_annotation_filename.replace("annotations", source_prediction_dirname)
                

                callbacks.append(functools.partial(
                    save_prediction,
                    filename=source_prediction_filename,
                    boxes_3d=source_pd_boxes_3d,
                    boxes_2d=source_pd_boxes_2d,
                ))

            averaged_iou_matrix = accumulated_iou_matrix / accumulated_cnt_matrix
            matched_pd_indices, matched_gt_indices = vsrd.utils.torch_function(sp.optimize.linear_sum_assignment)(averaged_iou_matrix, maximize=True)

            confidences = averaged_iou_matrix[matched_pd_indices, matched_gt_indices]

            for callback in callbacks:
                callback(confidences=confidences)
                        
def Re_Ordered_the_Dyanmic(target_input_extrinsic_matrices,
                           target_input_intrinsic_matrices,
                           target_inputs_boxes_2d,
                           world_boxes_3d,
                           original_dynamic_list):
    
    target_input_extrinsic_matrices = target_input_extrinsic_matrices.unsqueeze(0)
    target_input_intrinsic_matrices = target_input_intrinsic_matrices.unsqueeze(0)
    target_inputs_boxes_2d = target_inputs_boxes_2d.unsqueeze(0)
    
    camera_boxes_3d = torch.einsum("bmn,b...n->b...m", target_input_extrinsic_matrices, world_boxes_3d)
    camera_boxes_3d = camera_boxes_3d[..., :-1] / camera_boxes_3d[..., -1:] #(1,4,8,3)---> at source view camera coordinate

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
        in zip(camera_boxes_3d, target_input_intrinsic_matrices)
    ], dim=0) #(1,4,2,2)

    temp_images = torch.randn(1,3,376,1408)
    # make sure inside the image: This is the estimated 2D Bounding Boxes
    camera_boxes_2d = torchvision.ops.clip_boxes_to_image(
        boxes=camera_boxes_2d.flatten(-2, -1),
        size=temp_images.shape[-2:],
    ).unflatten(-1, (2, 2))  #(1,4,2,2)


    # bipartite_matching
    matching_cost_matrices = [
        -torchvision.ops.distance_box_iou(
            boxes1=pd_boxes_2d.flatten(-2, -1),
            boxes2=gt_boxes_2d.flatten(-2, -1),
        )
        for pd_boxes_2d, gt_boxes_2d
        in zip(camera_boxes_2d, target_inputs_boxes_2d)
        ]
    
    matched_indices = list(map(
        utils.torch_function(sp.optimize.linear_sum_assignment),
        matching_cost_matrices,
    )) 
    
    initial_estimated_matched_indices, initial_gt_matched_indices = matched_indices[0]
    initial_estimated_matched_indices_list = initial_estimated_matched_indices.cpu().numpy().tolist() # list for estimated 
    initial_gt_matched_indices_list = initial_gt_matched_indices.cpu().numpy().tolist() # list for GT
    
    revised_original_dynamic_list = calculate_framewise_dynamic_mask(
                    N=len(initial_gt_matched_indices_list),
                    target_frame_dynamic_list=original_dynamic_list,
                    target_frame_matched_indices=initial_gt_matched_indices_list,
                    source_frame_matched_indices=initial_estimated_matched_indices_list)
    
    return revised_original_dynamic_list

def calculate_framewise_dynamic_mask(N, target_frame_dynamic_list, target_frame_matched_indices, source_frame_matched_indices):
    B_Mask = [False] * N
    for i in range(N):
      A_index = target_frame_matched_indices[i]
      B_index = source_frame_matched_indices[i]
      Bool_value = target_frame_dynamic_list[A_index]
      B_Mask[B_index] = Bool_value
      
    return B_Mask

    
if __name__=="__main__":
    
    
    parser = argparse.ArgumentParser(description="VSRD++: Prediction Maker for KITTI-360")
    parser.add_argument("--root_dirname", type=str, default="/media/zliu/data12/dataset/VSRD_PP_Sync/")
    parser.add_argument("--ckpt_dirname", type=str, default="ckpts/kitti_360/vsrd")
    parser.add_argument("--ckpt_filename", type=str, default="step_2499.pt")
    parser.add_argument("--dyanmic_root_filename",type=str,default="None")
    parser.add_argument("--input_model_type",type=str,default="None",help="Selected from [vanilla,velocity,mlp,velocity_with_init]")
    parser.add_argument("--saved_pseudo_folder_path",type=str,default="predictions",
                        help="Selected from [vanilla,velocity,mlp,velocity_with_init]")
    
    parser.add_argument("--split_dirname", type=str, default="R50-N16-M128-B16")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--num_workers", type=int, default=9)
    args = parser.parse_args()

    main(args=args)

