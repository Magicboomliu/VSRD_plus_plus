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

import trimesh

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
from Optimized_Based.configs.train_config_sequence_00 import _C as my_conf_train
import vsrd.distributed
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from get_mesh import generate_mesh_from_sdf_network

def quadrature_sampler(bins, deterministic=False):
    weights = 0.5 if deterministic else torch.rand_like(bins[..., :-1])
    samples = torch.lerp(bins[..., :-1], bins[..., 1:], weights)
    return samples


def inverse_transform_sampler(bins, weights, num_samples, deterministic=False):

    pdf = nn.functional.normalize(weights, p=1, dim=-1)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = nn.functional.pad(cdf, (1, 0))

    if deterministic:
        uniform = torch.linspace(0.0, 1.0, num_samples, device=cdf.device)
        uniform = uniform.expand(*cdf.shape[:-1], -1)
    else:
        uniform = torch.rand(*cdf.shape[:-1], num_samples, device=cdf.device)
        uniform = torch.sort(uniform, dim=-1, descending=False).values

    indices = torch.searchsorted(cdf, uniform, right=False)
    indices = torch.clamp(indices, min=1, max=cdf.shape[-1] - 1)

    min_cdf = torch.gather(cdf, index=indices - 1, dim=-1)
    max_cdf = torch.gather(cdf, index=indices, dim=-1)

    min_bins = torch.gather(bins, index=indices - 1, dim=-1)
    max_bins = torch.gather(bins, index=indices, dim=-1)

    weights = (uniform - min_cdf) / (max_cdf - min_cdf + 1e-6)
    samples = torch.lerp(min_bins, max_bins, weights)

    return samples



def hierarchical_volumetric_rendering(
    distance_field,
    ray_positions,
    ray_directions,
    distance_range,
    num_samples,
    sdf_std_deviation,
    cosine_ratio=1.0,
    epsilon=1e-6,
    sampled_distances=None,
    sampled_weights=None,
):
    '''
    distance field: SDF generator 
    ray_position: [1000,3]
    ray_direction: [1000,3]
    distance_range : [0,100]
    sdf_std_deviation: 1.0
    
    
    '''
    

    
    
    # corase sampling
    if sampled_distances is None:
        distance_bins = torch.linspace(*distance_range, num_samples + 1, device=ray_directions.device) #[101]

        distance_bins = distance_bins.expand(*ray_directions.shape[:-1], 1, -1) #[1000,1,101]
        
        # quadrature_sampler 函数在每一对连续的区间端点之间生成采样点。这些采样点可以是确定的中间点，也可以是随机点。
        sampled_distances = quadrature_sampler(distance_bins) #[1000,1,100]

    # fine sampling
    else:
        sampled_distances = sampled_distances.permute(*range(1, sampled_distances.ndim), 0)
        sampled_weights = sampled_weights.permute(*range(1, sampled_weights.ndim), 0)

        sampled_distances = torch.cat([
            sampled_distances,
            inverse_transform_sampler(
                bins=sampled_distances,
                weights=sampled_weights,
                num_samples=num_samples,
            ),
        ], dim=-1)

        sampled_distances = torch.sort(sampled_distances, dim=-1, descending=False).values

    # 的意思是将最后一个维度（-1）移到最前面，其余维度依次后移。[100, 1000, 1]
     
    sampled_distances = sampled_distances.permute(-1, *range(sampled_distances.ndim - 1)) # Size([100, 1000, 1]) 
    sampled_intervals = sampled_distances[1:, ...] - sampled_distances[:-1, ...] # sample point intervals # size(99,1000,1)
    sampled_midpoints = (sampled_distances[:-1, ...] + sampled_distances[1:, ...]) / 2.0 # sample mid points # size(99,1000,1)

    sampled_positions = ray_positions + ray_directions * sampled_midpoints # all the smaple points direction # Size([99, 1000, 3]) 
       
    create_graph = torch.is_grad_enabled()
    

    with torch.enable_grad():
        sampled_positions.requires_grad_(True)
        sampled_signed_distances, *multi_sampled_features = distance_field(sampled_positions) # get sdf and surface normals


    return sampled_signed_distances


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

def before_with_2013_string(string):
    return string[:string.index("2013")]

def after_with_2013_string(string):
    return string[string.index("2013"):]


def changed_current_filename(string,relative_index):
    updated_string = int(string) + int(relative_index)
    
    updated_string = format_number(updated_string)
    return updated_string
    
def format_number(num, length=10):
    return str(num).zfill(length)



def Extract_Mesh_From_VSRDPP(args):

    # get the dataset names
    train_filename_list = my_conf_train.TRAIN.DATASET.FILENAMES
    class_names = my_conf_train.TRAIN.DATASET.CLASS_NAMES
    num_of_workers = my_conf_train.TRAIN.DATASET.NUMS_OF_WORKERS
    num_source_frames = my_conf_train.TRAIN.DATASET.NUM_SOURCE_FRAMES # 16 by default

    # Dataset Preprocessing
    target_transforms_resize_size = my_conf_train.TRAIN.DATASET.TARGET_TRANSFORMS.IMAGE_SIZE 
    target_transforms_min_mask_area1 = my_conf_train.TRAIN.DATASET.TARGET_TRANSFORMS.MIN_MASK_AREA_01
    target_transforms_min_mask_area2 = my_conf_train.TRAIN.DATASET.TARGET_TRANSFORMS.MIN_MASK_AREA_02
    target_transforms_min_box_size = my_conf_train.TRAIN.DATASET.TARGET_TRANSFORMS.MIN_BOX_SIZE
    
    source_transforms_resize_size = my_conf_train.TRAIN.DATASET.SOURCE_TRANSFORMS.IMAGE_SIZE 
    source_transforms_min_mask_area1 = my_conf_train.TRAIN.DATASET.SOURCE_TRANSFORMS.MIN_MASK_AREA_01
    source_transforms_min_mask_area2 = my_conf_train.TRAIN.DATASET.SOURCE_TRANSFORMS.MIN_MASK_AREA_02
    source_transforms_min_box_size = my_conf_train.TRAIN.DATASET.SOURCE_TRANSFORMS.MIN_BOX_SIZE
    
    dataset_rectification = my_conf_train.TRAIN.DATASET.RECTIFICATION
    
    
    # Using Dynamic Masks Or Not
    USE_DYNAMIC_MASK_FLAG = args.use_dynamic_mask
    # Using Dynamic Mask Modeling
    USE_DYNAMIC_MODELING_FLAG = args.dynamic_modeling
    # Using RDF or Not
    USE_RDF_MODELING_FLAG = args.use_residual_distance_field
    # without box flag
    WITHOUT_BOX_FLAG = args.without_box
    

    if USE_RDF_MODELING_FLAG:
        flag = 'With_RDF'
    else:
        flag = 'Without_RDF' 


    target_transforms=[Resizer(image_size=target_transforms_resize_size),
                           MaskAreaFilter(min_mask_area=target_transforms_min_mask_area1),
                           MaskRefiner(),
                           MaskAreaFilter(min_mask_area=target_transforms_min_mask_area2),
                           BoxGenerator(),
                           BoxSizeFilter(min_box_size=target_transforms_min_box_size),
                           SoftRasterizer()
                           ]
    
    source_transforms=[Resizer(image_size=source_transforms_resize_size),
                           MaskAreaFilter(min_mask_area=source_transforms_min_mask_area1),
                           MaskRefiner(),
                           MaskAreaFilter(min_mask_area=source_transforms_min_mask_area2),
                           BoxGenerator(),
                           BoxSizeFilter(min_box_size=source_transforms_min_box_size),
                           SoftRasterizer()]

    # ====================================================================================================
    # datasets
    dataset_kitti360 = KITTI360Dataset(filenames=train_filename_list,
                              class_names=class_names,
                              num_of_workers=num_of_workers,
                              num_source_frames=num_source_frames,
                              target_transforms=target_transforms,
                              source_transforms=source_transforms,
                              rectification=dataset_rectification)
    

    # ====================================================================================================
    # loaders
    loaders = DataLoader(dataset=dataset_kitti360,
                         batch_size=1,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=False,
                         collate_fn=collate_nested_dicts)
    
    device_id = 0
    device = device_id
    global_step = int((args.ckpt_filename)[5:-3])

    
    index = 0
    
    for multi_inputs in tqdm(loaders):

        #----------------------------------------------------------------------------------------------------------------#
            #-------------------------------- Loading the Data ----------------------------------------------------------#
        #----------------------------------------------------------------------------------------------------------------#
        multi_inputs = {
            relative_index: Dict.apply({
                key if re.fullmatch(r".*_\dd", key) else inflection.pluralize(key): value
                for key, value in inputs.items()
            })
            for relative_index, inputs in multi_inputs.items()}
        
        multi_inputs = utils.to(multi_inputs, device=device_id, non_blocking=True)
        target_inputs = multi_inputs[0] # target inputs
        # ================================================================
        # logging
        image_filename, = target_inputs.filenames    #/media/zliu/data12/dataset/KITTI/VSRD_Format/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000793.png
        root_dirname = dataset_kitti360.get_root_dirname(image_filename)  #/media/zliu/data12/dataset/KITTI/VSRD_Format
        image_dirname = os.path.splitext(os.path.relpath(image_filename, root_dirname))[0] #data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000793
        logger = utils.get_logger(image_dirname)
        

        #----------------------------------------------------------------------------------------------------------------#
        #-------------------------------- Loading the Dynamic Labels ------------------------------------------------#
        #----------------------------------------------------------------------------------------------------------------#

        dynamic_labels_for_target_view = dict()
        dynamic_labels_for_target_view["instance_ids"] = []
        dynamic_labels_for_target_view["dynamic_labels"] = []
        dynamic_raw_contents = read_text_lines(my_conf_train.TRAIN.DYNAMIC_LABELS_PATH)
        
        
        for content in dynamic_raw_contents:
            content = content.strip()
            current_returned_ids,current_returned_filename,current_return_labels = content.split(" ")
            
            if before_with_2013_string(current_returned_filename)!=before_with_2013_string(image_filename):
                current_returned_filename = current_returned_filename.replace(before_with_2013_string(current_returned_filename),before_with_2013_string(image_filename))

            if current_returned_filename == image_filename:
                dynamic_labels_for_target_view['instance_ids'] = current_returned_ids
                dynamic_labels_for_target_view["dynamic_labels"] = current_return_labels
                break


        dynamic_mask_for_target_view = dynamic_labels_for_target_view['dynamic_labels']
        dynamic_mask_for_target_view = [bool(int(float(data))) for data in dynamic_mask_for_target_view.split(",")]
        
        
        

        #----------------------------------------------------------------------------------------------------------------#
        #-------------------------------- Loading the Per-Trained Models ------------------------------------------------#
        #----------------------------------------------------------------------------------------------------------------#
        saved_ckpt_fname_folder = image_filename.replace(args.root_dirname,args.ckpt_dirname)[:-4]
        saved_ckpt_fname_name = os.path.join(saved_ckpt_fname_folder,args.ckpt_filename)
        assert os.path.exists(saved_ckpt_fname_name)
        
        target_checkpoint = torch.load(saved_ckpt_fname_name, map_location="cpu")
        num_instances = target_checkpoint["models"]["detector"]["embeddings"].shape[1]
        models_weights_new = target_checkpoint['models']


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
        print("Loaded all the pre-trained Models for Name {}.".format(saved_ckpt_fname_name))


        #----------------------------------------------------------------------------------------------------------------#
        #-------------------------------- Data Alignment with Target Views ------------------------------------------------#
        #----------------------------------------------------------------------------------------------------------------#
        for source_inputs in multi_inputs.values():
            source_instance_indices = [] 
            for source_instance_ids, target_instance_ids in zip(source_inputs.instance_ids, target_inputs.instance_ids):
                indices = [
                    source_instance_ids.tolist().index(target_instance_id.item()) 
                    if target_instance_id in source_instance_ids else -1 
                    for target_instance_id in target_instance_ids]
            
                source_instance_indices.append(source_instance_ids.new_tensor(indices))
            
            
            # instance by instance 
            source_labels = [
                utils.reversed_pad(source_labels, (0, 1))[source_instance_indices, ...]
                for source_labels, source_instance_indices
                in zip(source_inputs.labels, source_instance_indices)
            ] 
            
            source_boxes_2d = [
                utils.reversed_pad(source_boxes_2d, (0, 1))[source_instance_indices, ...]
                for source_boxes_2d, source_instance_indices
                in zip(source_inputs.boxes_2d, source_instance_indices)
            ]
            
            source_boxes_3d = [
                utils.reversed_pad(source_boxes_3d, (0, 1))[source_instance_indices, ...]
                for source_boxes_3d, source_instance_indices
                in zip(source_inputs.boxes_3d, source_instance_indices)
            ] 
            
            source_hard_masks = [
                utils.reversed_pad(source_masks, (0, 1))[source_instance_indices, ...]
                for source_masks, source_instance_indices
                in zip(source_inputs.hard_masks, source_instance_indices)
            ]

            source_soft_masks = [
                utils.reversed_pad(source_soft_masks, (0, 1))[source_instance_indices, ...]
                for source_soft_masks, source_instance_indices
                in zip(source_inputs.soft_masks, source_instance_indices)
            ]

            source_instance_ids = [
                utils.reversed_pad(source_instance_ids, (0, 1))[source_instance_indices, ...]
                for source_instance_ids, source_instance_indices
                in zip(source_inputs.instance_ids, source_instance_indices)
            ]

            source_visible_masks = [
                source_instance_indices.cpu() >= 0
                for source_instance_indices in source_instance_indices
            ] #[tensor([ True,  True,  True,  True, False,  True])]

            source_inputs.update(
                labels=source_labels,
                boxes_2d=source_boxes_2d,
                boxes_3d=source_boxes_3d,
                hard_masks=source_hard_masks,
                soft_masks=source_soft_masks,
                instance_ids=source_instance_ids,
                visible_masks=source_visible_masks,
            )
                            
        #----------------------------------------------------------------------------------------------------------------#
        #-------------------------------- Get Rendering Sampling Points ------------------------------------------------#
        #----------------------------------------------------------------------------------------------------------------#
        for inputs in multi_inputs.values():
            camera_positions, ray_directions = ray_casting(
                image_size=inputs.images.shape[-2:],
                intrinsic_matrices=inputs.intrinsic_matrices,
                extrinsic_matrices=inputs.extrinsic_matrices,
            )
            inputs.update(
                ray_directions=ray_directions,
                camera_positions=camera_positions,
            )
            
        # multi_hard_masks
        multi_hard_masks = list(map(torch.stack, zip(*[
            list(map(
                functools.partial(torch.permute, dims=(1, 2, 0)),
                inputs.hard_masks,
            ))
            for inputs in multi_inputs.values()
        ]))) # [torch.Size([nums_of_frames, H, W, nums_of_instances])]

        multi_images = list(map(torch.stack, zip(*[
            list(map(
                functools.partial(torch.permute, dims=(1, 2, 0)),
                inputs.images,
            ))
            for inputs in multi_inputs.values()
        ]))) # [torch.Size([nums_of_frames, H, W, nums_of_image_channels])]
        # ================================================================


        nums_of_source_images_integrated_into_rendering = multi_images[0].shape[0]
        device = multi_images[0].device
        nums_of_instance_number = multi_hard_masks[0].shape[-1]

        # save images
        current_scene_name = multi_inputs[0]['filenames'][0]
        basename_filename = after_with_2013_string(current_scene_name)
        updated_image_frame_idx = basename_filename
        current_basename_filename = os.path.join(os.path.dirname(current_scene_name),flag,updated_image_frame_idx+".png")
        current_basename_filename = after_with_2013_string(current_basename_filename)
        saved_image_filename = os.path.join(args.output_folder,current_basename_filename)
        saved_image_dirname = os.path.dirname(saved_image_filename)
        os.makedirs(saved_image_dirname,exist_ok=True)



        # ----------------------------------------------------------------
        # instance loss
        cosine_annealing = lambda x, a, b: (np.cos(np.pi * x) + 1.0) / 2.0 * (a - b) + b
        cosine_ratio = global_step / my_conf_train.TRAIN.OPTIMIZATION_NUM_STEPS
        
        sdf_union_temperature = cosine_annealing(
            global_step / my_conf_train.TRAIN.OPTIMIZATION_NUM_STEPS,
            my_conf_train.TRAIN.VOLUME_RENDERING.MAX_SDF_UNION_TEMPERATURE,
            my_conf_train.TRAIN.VOLUME_RENDERING.MIN_SDF_UNION_TEMPERATURE,
        )
        sdf_std_deviation = cosine_annealing(
            global_step / my_conf_train.TRAIN.OPTIMIZATION_NUM_STEPS,
            my_conf_train.TRAIN.VOLUME_RENDERING.MAX_SDF_STD_DEVIATION,
            my_conf_train.TRAIN.VOLUME_RENDERING.MIN_SDF_STD_DEVIATION)
        
        # RDF
        def residual_distance_field(distance_field):
            def wrapper(positions):
                # # 分离输入位置数据的 x、y、z 维度
                x_positions, y_positions, z_positions = torch.unbind(positions, dim=-1)
                
                # # 处理 x 维度，将其取绝对值，然后重新组合成位置数据
                positions = torch.stack([torch.abs(x_positions), y_positions, z_positions], dim=-1)
                # positions = torch.tanh(positions / torch.max(models.detector.dimension_range, dim=0).values)
                positions = positions / max(my_conf_train.TRAIN.VOLUME_RENDERING.DISTANCE_RANGE)
                positions = models.positional_encoder(positions)
                distances = distance_field(positions)
                ## 在训练过程中，输出的值需要与目标值进行比较（通常通过损失函数）。
                # 减去1.0可以使得距离值更好地适应损失函数的计算，使得梯度更稳定，从而有助于模型的训练。
                distances = torch.sigmoid(distances - 1.0) 
                return distances
            return wrapper

        def residual_composition(distance_field, residual_distance_field):
            def wrapper(positions):
                distances = distance_field(positions)
                residuals = residual_distance_field(positions)
                return distances + residuals

            return wrapper

        def instance_field(distance_field, instance_label):
            def wrapper(positions):
                distances = distance_field(positions) # get the SDF
                # positions = torch.tanh(positions / torch.max(models.detector.dimension_range, dim=0).values)
                positions = positions / max(my_conf_train.TRAIN.VOLUME_RENDERING.DISTANCE_RANGE) # SDF Normaliozatozn
                positions = models.positional_encoder(positions)

                instance_labels = nn.functional.one_hot(instance_label, num_instances)
                instance_labels = instance_labels.expand(*distances.shape[:-1], -1)

                return distances, instance_labels
            return wrapper
        
        # Final SDF is all SDF softmin, here the distance field i
        def soft_union(distance_fields, temperature):
            def wrapper(positions):
                
                distances, *multi_features = map(torch.stack, zip(*[
                    distance_field(positions)
                    for distance_field in distance_fields
                ]))
        
                weights = nn.functional.softmin(distances / temperature, dim=0)
                distances = torch.sum(distances * weights, dim=0)
                multi_features = [
                    torch.sum(features * weights, dim=0)
                    for features in multi_features
                ]
                return distances, *multi_features

            return wrapper

        # Use the one with the smallest SDF
        def hard_union(distance_fields):
            def wrapper(positions):
                distances, *multi_features = map(torch.stack, zip(*[
                    distance_field(positions)
                    for distance_field in distance_fields
                ]))
                indices = torch.argmin(distances, dim=0, keepdim=True)
                distances = torch.gather(distances, index=indices, dim=0).squeeze(0)
                multi_features = [
                    torch.gather(features, index=indices.expand(*indices.shape[:-1], *features.shape[-1:]), dim=0).squeeze(0)
                    for features in multi_features
                ]
                return distances, *multi_features

            return wrapper

        def hierarchical_wrapper(renderer):

            def wrapper(*args, **kwargs):

                # 使用torch.no_grad()上下文管理器，防止在推理过程中计算梯度
                with torch.no_grad():
                    # 调用渲染器，获取多个返回值，包括采样距离和采样权重
                    *_, sampled_distances, sampled_weights = renderer(*args, **kwargs)
                # 更新关键字参数，将采样距离和采样权重传递回渲染器
                kwargs.update(sampled_distances=sampled_distances, sampled_weights=sampled_weights)
                # 再次调用渲染器，这次使用更新后的关键字参数
                *outputs, _, _ = renderer(*args, **kwargs)
                # 返回渲染器的输出值，不包括采样距离和采样权重
                return outputs

            return wrapper


        with torch.no_grad():                
            #['boxes_3d', 'locations', 'dimensions', 'orientations', 'embeddings'] 
            world_outputs = utils.Dict.apply(models.detector())
            multi_outputs = utils.DefaultDict(utils.Dict)
            # This is the shared base 3D Bounding Boxes
            world_boxes_3d = nn.functional.pad(world_outputs.boxes_3d, (0, 1), mode="constant", value=1.0) # (1,num_of_instances,8,4)
            relative_index_list = [relative_index for relative_index in multi_inputs.keys()]

            if USE_DYNAMIC_MASK_FLAG:
                current_velocity = models['detector'].velocity #[1,8,3]
                current_velocity = current_velocity.unsqueeze(-2)
                relative_box_residual_list = [current_velocity *t for t in relative_index_list]
                relative_box_residual = torch.cat(relative_box_residual_list,dim=-2)  
            else:
                relative_box_residual = torch.zeros((1,nums_of_instance_number,nums_of_source_images_integrated_into_rendering,3)).to(device_id)

            # Align the dynamic mask with the current output.                            
            dynamic_mask_for_target_view_for_output = get_dynamic_mask_for_the_world_output(target_inputs=target_inputs,
                                                        world_boxes_3d=world_boxes_3d,
                                                            dynamic_mask_for_target_view=dynamic_mask_for_target_view)

            # Use the Residual MOdeling
            if USE_RDF_MODELING_FLAG:
                # #(1,4,1617), here the 1617 is the unit numbers.
                distance_field_weights = models.hyper_distance_field(world_outputs.embeddings)             
                world_outputs.update(distance_field_weights=distance_field_weights)

                soft_distance_fields = []
                
                # for all the frames for one scene
                for locations, dimensions, orientations, distance_field_weights in zip(
                    world_outputs.locations,
                    world_outputs.dimensions,
                    world_outputs.orientations,
                    world_outputs.distance_field_weights):
                    
                    
                    # 初始化一个列表来存储每个实例的距离场
                    distance_fields = []
                    # 遍历场景中的每个实例，提取位置、尺寸、方向和距离场权重，并生成实例标签
                    for instance_label, (location, dimension, orientation, distance_field_weights) in enumerate(zip(locations,dimensions,orientations,distance_field_weights)):
                        
                        if USE_DYNAMIC_MASK_FLAG:
                            if dynamic_mask_for_target_view_for_output[instance_label]:
                                # dynamic objects
                                instance_wise_box_residual = relative_box_residual[:,instance_label,:,:] # [1,16,3]
                                
                            else:
                                # static objects
                                current_relative_box_residual = torch.zeros_like(relative_box_residual)
                                instance_wise_box_residual = current_relative_box_residual[:,instance_label,:,:]
                            
                        else:
                            if USE_DYNAMIC_MODELING_FLAG:
                                instance_wise_box_residual = relative_box_residual[:,instance_label,:,:] # [1,16,3]
                            else:
                                # FIXME
                                instance_wise_box_residual = torch.zeros((1,nums_of_source_images_integrated_into_rendering,3)).to(device)


                        initial_distance_field_per_instance_before_union_list = []
                        base_distance_field = rendering.sdfs.box(dimension)
                        
                        # 使用residual_distance_field包裹距离场函数
                        residual_field = residual_distance_field(
                            distance_field=functools.partial(
                                models.hyper_distance_field.distance_field,
                                distance_field_weights,
                            ),)

                        # 组合基础距离场和残差距离场
                        combined_field = residual_composition(
                                distance_field=base_distance_field,
                                residual_distance_field=residual_field)
                        
                        # 应用实例标签
                        instance_field_with_label = instance_field(
                            distance_field=combined_field,
                            instance_label=dimension.new_tensor(instance_label, dtype=torch.long))

                        # 应用旋转变换
                        rotated_field = rendering.sdfs.rotation(
                            instance_field_with_label,
                            orientation)


                        # Here multiple across the the image frames
                        for current_frame_idx in range(len(multi_inputs.keys())):
                            current_location_residual = instance_wise_box_residual[:,current_frame_idx,:]
                            translated_field = rendering.sdfs.translation(
                                                    rotated_field,
                                                    location + current_location_residual)         
                            # M's translation field
                            initial_distance_field_per_instance_before_union_list.append(translated_field)
                            
                        # 将处理后的距离场添加到实例列表中
                        distance_fields.append(initial_distance_field_per_instance_before_union_list)


                    # scene union
                    for current_frame_idx in range(len(multi_inputs.keys())):
                        current_distance_fields = [d[current_frame_idx] for d in distance_fields] # get current distance fields
                        # 将所有实例的距离场进行软联合
                        soft_distance_field = soft_union(
                            distance_fields=current_distance_fields,
                            temperature=sdf_union_temperature)
                        # 将处理后的软距离场添加到场景列表中: length should be eight
                        soft_distance_fields.append(soft_distance_field)

            else:
                if USE_DYNAMIC_MODELING_FLAG:
                    boxes_residuals = relative_box_residual
                else:
                    boxes_residuals = torch.zeros(1,8,nums_of_source_images_integrated_into_rendering,3).to(device)

                # 初始化一个列表来存储软距离场
                soft_distance_fields = []
                # 遍历每个场景的输出，提取位置、尺寸和方向
                for locations, dimensions, orientations in zip(
                    world_outputs.locations,
                    world_outputs.dimensions,
                    world_outputs.orientations,
                ):
                    # 初始化一个列表来存储每个实例的距离场
                    distance_fields = []
                    # 遍历场景中的每个实例，提取位置、尺寸和方向，并生成实例标签
                    for instance_label, (location, dimension, orientation) in enumerate(zip(locations,dimensions,orientations)):
                        
                        if USE_DYNAMIC_MASK_FLAG:
                            if dynamic_mask_for_target_view_for_output[instance_label]:
                                # dynamic objects
                                instance_wise_box_residual = boxes_residuals[:,instance_label,:,:] # [1,16,3]
                                
                            else:
                                # static objects
                                current_boxes_residuals = torch.zeros_like(boxes_residuals)
                                instance_wise_box_residual = current_boxes_residuals[:,instance_label,:,:] # [1,16,3]
                        else:
                            instance_wise_box_residual = boxes_residuals[:,instance_label,:,:] # [1,16,3]
                                
                            
                        initial_distance_field_per_instance_before_union_list = []
                        # 构建基础的盒子距离场
                        base_distance_field = rendering.sdfs.box(dimension)
                        # 创建带有实例标签的距离场
                        instance_field_with_label = instance_field(
                            distance_field=base_distance_field,
                            instance_label=dimension.new_tensor(instance_label, dtype=torch.long)
                        )
                        rotated_field = rendering.sdfs.rotation(
                            instance_field_with_label,
                            orientation)

                        # Here multiple across the the image frames
                        for current_frame_idx in range(len(multi_inputs.keys())):
                            current_location_residual = instance_wise_box_residual[:,current_frame_idx,:]
                            # 应用平移变换
                            translated_field = rendering.sdfs.translation(
                                                    rotated_field,
                                                    location + current_location_residual)         
                            # M's translation field
                            initial_distance_field_per_instance_before_union_list.append(translated_field)

                        # 将处理后的距离场添加到实例列表中
                        distance_fields.append(initial_distance_field_per_instance_before_union_list)

                    # 将所有实例的距离场进行软联合
                    for current_frame_idx in range(len(multi_inputs.keys())):
                        current_distance_fields = [d[current_frame_idx] for d in distance_fields] # get current distance fields
                        # 将所有实例的距离场进行软联合
                        soft_distance_field = soft_union(
                            distance_fields=current_distance_fields,
                            temperature=sdf_union_temperature)
                        
                        # 将处理后的软距离场添加到场景列表中
                        soft_distance_fields.append(soft_distance_field)

            
            for frame_index,(relative_index, inputs) in enumerate(multi_inputs.items()):
                
                basename_filename = after_with_2013_string(inputs['filenames'][0])
                image_frame_idx = os.path.basename(basename_filename)[:-4]
                
                # FIXME ME
                updated_image_frame_idx = changed_current_filename(string=image_frame_idx,
                                         relative_index=relative_index)
                            
                distance_field=soft_distance_fields[frame_index]
                verts, faces = generate_mesh_from_sdf_network(distance_field)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                mesh.export("{}/{}.ply".format(saved_image_dirname,updated_image_frame_idx))




if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="VSRD++: Prediction Maker for KITTI-360")
    parser.add_argument("--root_dirname", type=str, default="/media/zliu/data12/dataset/VSRD_PP_Sync/")
    parser.add_argument("--ckpt_dirname", type=str, default="/media/zliu/data12/dataset/TPAMI_Saved_Ckpts/VSRDPP_saved_ckpts/ckpts/")
    parser.add_argument("--ckpt_filename", type=str, default="step_2499.pt")
    parser.add_argument(
        "--use_dynamic_mask", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dynamic_modeling", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    
    parser.add_argument(
        "--use_residual_distance_field", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")

    parser.add_argument(
        "--without_box", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")


    parser.add_argument(
        "--output_folder", type=str, default="")
    
    args = parser.parse_args()
     
    

    Extract_Mesh_From_VSRDPP(args=args)
            
            
