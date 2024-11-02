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
import torch.nn.functional as F
import torchvision

import numpy as np
import scipy as sp
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import copy
import inflection
import torch.utils.tensorboard

import sys
sys.path.append("..")
# VSRD Dataset
from vsrd.utils import collate_nested_dicts,Dict,ProgressMeter,StopWatch
from vsrd import utils
from torch.utils.data import Dataset, DataLoader
from vsrd.datasets.kitti360_dataset import KITTI360Dataset
from vsrd.datasets.transforms import Resizer,MaskAreaFilter,MaskRefiner,BoxGenerator,BoxSizeFilter,SoftRasterizer

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
from tqdm import tqdm
import datetime
from vsrd import visualization

from Optimized_Based.utils.box_geo import decode_box_3d,divide_into_n_parts,get_dynamic_mask_for_the_world_output,get_dynamic_sequentail_for_the_world_output
from Optimized_Based.utils.file_io import read_text_lines,read_complex_strings
import re

import vsrd
import vsrd.distributed
import multiprocessing
from vsrd.distributed.loader import DistributedDataLoader

# Initialization
import argparse


import pickle
def read_pickle_file(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

LINE_INDICES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]



def encode_location(decoded_locations):
    # Sync into the Same Device 
    current_device = decoded_locations.device

    location_range=[
        [-50.0, 1.55 - 1.75 / 2.0 - 5.0, 000.0],
        [+50.0, 1.55 - 1.75 / 2.0 + 5.0, 100.0]]

    location_range = torch.as_tensor(location_range).to(current_device)
    low, high = location_range
    low = low.clone().detach().to(decoded_locations.device)
    high = high.clone().detach().to(decoded_locations.device)
    
    decoded_locations = torch.clamp(decoded_locations, min=low, max=high)
    
    encoded_locations = torch.logit((decoded_locations - low) / (high - low).clamp(min=1e-6))
    
    # 使用 torch.where 将 -inf 替换为 -4
    encoded_locations = torch.where(torch.isneginf(encoded_locations), torch.tensor(-4.0).to(encoded_locations.device), encoded_locations)
    return encoded_locations


def encode_orientation(rotation_matrices):
    # 提取 rotation_matrices 中的 cos 和 sin 值
    cos = rotation_matrices[..., 0, 0]
    sin = rotation_matrices[..., 0, 2]
    encoded_orientations = torch.stack([cos, sin], dim=-1)
    return nn.functional.normalize(encoded_orientations, dim=-1)



def main(args=None):
    
    if args.config_path=='00':
        from Optimized_Based.configs.train_config_sequence_00 import _C as my_conf_train
    if args.config_path=='02':
        from Optimized_Based.configs.train_config_sequence_02 import _C as my_conf_train
    if args.config_path=='03':
        from Optimized_Based.configs.train_config_sequence_03 import _C as my_conf_train
    if args.config_path=='04':
        from Optimized_Based.configs.train_config_sequence_04 import _C as my_conf_train
    if args.config_path=='05':
        from Optimized_Based.configs.train_config_sequence_05 import _C as my_conf_train
    if args.config_path=='06':
        from Optimized_Based.configs.train_config_sequence_06 import _C as my_conf_train
    if args.config_path=='07':
        from Optimized_Based.configs.train_config_sequence_07 import _C as my_conf_train
    if args.config_path=='09':
        from Optimized_Based.configs.train_config_sequence_09 import _C as my_conf_train
    if args.config_path=='10':
        from Optimized_Based.configs.train_config_sequence_10 import _C as my_conf_train
    if args.config_path=='test':
        from Optimized_Based.configs.train_config_ddp_debug import _C as my_conf_train
    
    
    
    
    # DDP Settings
    # configuration
    if my_conf_train.TRAIN.DDP.LAUNCHER == "slurm":
        # NOTE: we must specify `MASTER_ADDR` and `MASTER_PORT` by the environment variables
        vsrd.distributed.init_process_group(backend=my_conf_train.TRAIN.DDP.BACKEND, port=my_conf_train.TRAIN.DDP.PORT)
    if my_conf_train.TRAIN.DDP.LAUNCHER == "torchrun":
        torch.distributed.init_process_group(backend=my_conf_train.TRAIN.DDP.BACKEND)
    device_id = vsrd.distributed.get_device_id(my_conf_train.TRAIN.DDP.NUM_DEVICES_PER_PROCESS, args.device_id)

    for rank in range(torch.distributed.get_world_size()):
        with vsrd.distributed.barrier():
            if torch.distributed.get_rank() == rank:
                world_size = torch.distributed.get_world_size()
                print(f"Rank: [{rank}/{world_size}] Device ID: {device_id}")
    
    
    # ================================================================
    # multiprocessing
    multiprocessing.set_start_method(my_conf_train.TRAIN.MULTIPROCESSING.START_METHOD, force=True)

    
    # reproducibility
    seed = my_conf_train.TRAIN.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
    
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
    train_batch_size = my_conf_train.TRAIN.DATASET.BATCH_SIZE # 1 by default


    loss_weight_list = {"eikonal_loss":my_conf_train.TRAIN.LOSS_WEIGHT.EIKONAL_LOSS,
                        "iou_projection_loss":my_conf_train.TRAIN.LOSS_WEIGHT.IOU_PROJECTION_LOSS,
                        "l1_projection_loss":my_conf_train.TRAIN.LOSS_WEIGHT.L1_PROJECTION_LOSS,
                        "photometric_loss":my_conf_train.TRAIN.LOSS_WEIGHT.PHOTOMETRIC_LOSS,
                        "radiance_loss":my_conf_train.TRAIN.LOSS_WEIGHT.RADIANCE_LOSS,
                        "silhouette_loss":my_conf_train.TRAIN.LOSS_WEIGHT.SILHOUETTE_LOSS}


    param_group_names = [
            "detector/locations",
            "detector/dimensions",
            "detector/orientations",
            "detector/embeddings",
            "hyper_distance_field"
        ]

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
    datasets = KITTI360Dataset(filenames=train_filename_list,
                              class_names=class_names,
                              num_of_workers=num_of_workers,
                              num_source_frames=num_source_frames,
                              target_transforms=target_transforms,
                              source_transforms=source_transforms,
                              rectification=dataset_rectification)

    # ====================================================================================================
    # loaders
    loaders = DistributedDataLoader(datasets, batch_size=train_batch_size, collate_fn=collate_nested_dicts,
                                    drop_last=True,
                                    pin_memory=False
                                    )
    # ====================================================================================================
    # utilities
    meters = Dict({
        "train": ProgressMeter(len(loaders) * my_conf_train.TRAIN.OPTIMIZATION_NUM_STEPS)
    })
    stop_watch = StopWatch()

    # Using Dynamic Masks Or Not
    USE_DYNAMIC_MODELING_FLAG = my_conf_train.TRAIN.USE_DYNAMIC_MODELING
    
    USE_DYNAMIC_MASK_FLAG = my_conf_train.TRAIN.USE_DYNAMIC_MASK
    DYNAMIC_TYPE = my_conf_train.TRAIN.DYNAMIC_MODELING_TYPE
    USE_RDF_MODELING_FLAG = my_conf_train.TRAIN.USE_RDF_MODELING

    
    def train():
        stop_watch.start()

        for multi_inputs in vsrd.distributed.tqdm(loaders):
            
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
            root_dirname = datasets.get_root_dirname(image_filename)  #/media/zliu/data12/dataset/KITTI/VSRD_Format
            image_dirname = os.path.splitext(os.path.relpath(image_filename, root_dirname))[0] #data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000793
            logger = utils.get_logger(image_dirname)
            

            # load the dynamic mask
            dynamic_labels_for_target_view = dict()
            dynamic_labels_for_target_view["instance_ids"] = []
            dynamic_labels_for_target_view["dynamic_labels"] = []
            
            if USE_DYNAMIC_MODELING_FLAG:
                if USE_DYNAMIC_MASK_FLAG:
                    dynamic_raw_contents = read_text_lines(my_conf_train.TRAIN.DYNAMIC_LABELS_PATH)
                    for content in dynamic_raw_contents:
                        content = content.strip()
                        current_returned_ids,current_returned_filename,current_return_labels = content.split(" ")
                        if current_returned_filename == image_filename:
                            dynamic_labels_for_target_view['instance_ids'] = current_returned_ids
                            dynamic_labels_for_target_view["dynamic_labels"] = current_return_labels
                            break
            
            
            # Output Locations
            ckpt_dirname = os.path.join(my_conf_train.TRAIN.CONFIG.replace("configs", "ckpts/{}".format(my_conf_train.TRAIN.MODEL_TYPE)),image_dirname)
            log_dirname = os.path.join(my_conf_train.TRAIN.CONFIG.replace("configs", "logs"),image_dirname)
            out_dirname = os.path.join(my_conf_train.TRAIN.CONFIG.replace("configs", "outs"),image_dirname)
            if os.path.exists(os.path.join(ckpt_dirname, f"step_{my_conf_train.TRAIN.OPTIMIZATION_NUM_STEPS - 1}.pt")):
                logger.warning(f"[{image_filename}] Already optimized. Skip this sample.")
                continue

            os.makedirs(log_dirname, exist_ok=True)
            log_filename = os.path.join(log_dirname, "log.txt")
            file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(levelname)s: %(asctime)s: %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # NOTE: store the main script and config for reproducibility
            shutil.copy(__file__, os.path.join(log_dirname, os.path.basename(__file__)))
            

            # check the number of instances
            num_instances, = map(len, target_inputs.hard_masks)
            if not num_instances:
                logger.warning(f"[{image_filename}] No instances. Skip this sample.")
                continue
            else:
                print("Founded Instances to be optimized, where the instance nums is {}".format(num_instances))
            
            
            # ================================================================
            # configuration
            logger.info(f"========== Multi-View Auto-Labeling Start ===================================")    
            logger.info("Datasets length: {}".format(len(datasets)))
            
            # ================================================================
            # define models: which is a combination of different modules.
            models = Dict()
            
            # VSRD++
            if USE_DYNAMIC_MODELING_FLAG:
                if DYNAMIC_TYPE =='mlp':
                    detector = BoxParameters3DRBN(batch_size=1,
                                       num_instances=num_instances,
                                       num_features=256)
                    # box residual 
                    box_residual_detector = ResidualBoxPredictor()
                elif DYNAMIC_TYPE =='vector_velocity':
                    detector = BoxParameters3D_With_Velocity(batch_size=1,
                                       num_instances=num_instances,
                                       num_features=256)
                elif DYNAMIC_TYPE == 'scalar_velocity':
                    detector = BoxParameters3D_With_Scalar_Velocity(batch_size=1,
                                       num_instances=num_instances,
                                       num_features=256)
                else:
                    raise NotImplementedError
            
            # Vanilla VSRD
            else:
                # box sdf
                detector = BoxParameters3D(batch_size=1,
                                       num_instances=num_instances,
                                       num_features=256)
            
            
            if USE_RDF_MODELING_FLAG:
                # hypernetwork
                hyper_distance_field = HyperDistanceField(in_channels=48,
                                                        out_channels_list=[16,16,16,16],
                                                        hyper_in_channels=256,
                                                        hyper_out_channels_list=[256,256,256,256])
            # positional_encoder
            positional_encoder = SinusoidalEncoder(num_frequencies=8)
            models['detector'] = detector
            if USE_RDF_MODELING_FLAG:
                models['hyper_distance_field'] = hyper_distance_field
                
            models['positional_encoder'] = positional_encoder
            if USE_DYNAMIC_MODELING_FLAG and  DYNAMIC_TYPE=='mlp':
                models['detector_residual'] = box_residual_detector  # Using MLP For Learning
            for model in models.values():
                model.to(device_id)
                
            logger.info(f"Models: {models}")
            
        
        
            '''Data Alignment with Target Views'''
            for source_inputs in multi_inputs.values():
                
                source_instance_indices = [] 

                for source_instance_ids, target_instance_ids in zip(source_inputs.instance_ids, target_inputs.instance_ids):
                    
                    indices = [
                        source_instance_ids.tolist().index(target_instance_id.item()) 
                        if target_instance_id in source_instance_ids else -1 
                        for target_instance_id in target_instance_ids
                    ]
               
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
                
            # Read the Dynamic Masks
            if USE_DYNAMIC_MODELING_FLAG:
                if USE_DYNAMIC_MASK_FLAG:
                    dynamic_mask_for_target_view = dynamic_labels_for_target_view['dynamic_labels']
                    
                    dynamic_mask_for_target_view = [bool(int(float(data))) for data in dynamic_mask_for_target_view.split(",")]
            
                        
            # ================================================================
            # optimizer
            if USE_DYNAMIC_MODELING_FLAG and  DYNAMIC_TYPE=='mlp':
                optimizer = optim.Adam([
                    {'params': models.detector.locations, 'lr': 0.01},
                    {'params': models.detector.dimensions, 'lr': 0.01},
                    {'params': models.detector.orientations, 'lr': 0.01},
                    {'params': models.detector.embeddings, 'lr': 0.001},
                    {'params': models.detector_residual.parameters(),'lr':0.00005},
                    {'params': models.hyper_distance_field.parameters(), 'lr': 0.0001}
                ], lr=0.01)  
                
            elif USE_DYNAMIC_MODELING_FLAG and DYNAMIC_TYPE=='vector_velocity':
                optimizer = optim.Adam([
                    {'params': models.detector.locations, 'lr': 0.01},
                    {'params': models.detector.dimensions, 'lr': 0.01},
                    {'params': models.detector.orientations, 'lr': 0.01},
                    {'params': models.detector.embeddings, 'lr': 0.001},
                    {'params': models.detector.velocity,'lr':0.005},
                    {'params': models.hyper_distance_field.parameters(), 'lr': 0.0001}
                ], lr=0.01)
                

            
            elif USE_DYNAMIC_MODELING_FLAG and DYNAMIC_TYPE=="scalar_velocity":
                optimizer = optim.Adam([
                    {'params': models.detector.locations, 'lr': 0.01},
                    {'params': models.detector.dimensions, 'lr': 0.01},
                    {'params': models.detector.orientations, 'lr': 0.01},
                    {'params': models.detector.embeddings, 'lr': 0.001},
                    {'params': models.detector.scalar_velocity,'lr':0.005},
                    {'params': models.hyper_distance_field.parameters(), 'lr': 0.0001}
                ], lr=0.01) 
            
            else:
                optimizer = optim.Adam([
                    {'params': models.detector.locations, 'lr': 0.01},
                    {'params': models.detector.dimensions, 'lr': 0.01},
                    {'params': models.detector.orientations, 'lr': 0.01},
                    {'params': models.detector.embeddings, 'lr': 0.001},
                    {'params': models.hyper_distance_field.parameters(), 'lr': 0.0001}
                ], lr=0.01) 


            # ================================================================
            # LR scheduler
            gamma = 0.01 ** (1.0 / 3000.0)
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            # ================================================================
            # summary writer
            writer = torch.utils.tensorboard.SummaryWriter(log_dirname)
            # ================================================================
            # checkpoint saver
            saver = utils.Saver(ckpt_dirname)
            # ================================================================
            


            # Prepared for Ray Sampling for all the images in the world space, which is also the target frame 0 recified space.
            for inputs in multi_inputs.values():
                camera_positions, ray_directions = ray_casting(
                    image_size=inputs.images.shape[-2:],
                    intrinsic_matrices=inputs.intrinsic_matrices,
                    extrinsic_matrices=inputs.extrinsic_matrices,
                )
                # camera_positions---> [1,3]
                # ray directions -----> [1,H,W,3]                
                inputs.update(
                    ray_directions=ray_directions,
                    camera_positions=camera_positions,
                )

            # gather all the source views
            multi_ray_directions = list(map(torch.stack, zip(*[
                inputs.ray_directions
                for inputs in multi_inputs.values()
            ]))) # torch.Size([num_of_frames, 376, 1408, 3]
            
            multi_camera_positions = list(map(torch.stack, zip(*[
                torch.stack(list(map(
                    torch.Tensor.expand_as,
                    inputs.camera_positions,
                    inputs.ray_directions,
                )), dim=0)
                for inputs in multi_inputs.values()
            ]))) # torch.Size([num_of_frames, 376, 1408, 3]
            
            # multi_hard_masks
            multi_hard_masks = list(map(torch.stack, zip(*[
                list(map(
                    functools.partial(torch.permute, dims=(1, 2, 0)),
                    inputs.hard_masks,
                ))
                for inputs in multi_inputs.values()
            ]))) # [torch.Size([nums_of_frames, H, W, nums_of_instances])]
            
            # I guess use for sampling
            multi_soft_masks = list(map(torch.stack, zip(*[
                list(map(
                    functools.partial(torch.permute, dims=(1, 2, 0)),
                    inputs.soft_masks,
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
            
            
            # training
            with utils.TrainSwitcher(*models.values()):
                for step in vsrd.distributed.tqdm(range(my_conf_train.TRAIN.OPTIMIZATION_NUM_STEPS), leave=False):
                    # ----------------------------------------------------------------
                    # inference here
                    with torch.enable_grad(): 
                        optimizer.zero_grad()     
                        world_outputs = utils.Dict.apply(models.detector()) #['boxes_3d', 'locations', 'dimensions', 'orientations', 'embeddings']               

            
                        # Compute the Box Residual: If Using the Dynamic Modeling
                        if step>=my_conf_train.TRAIN.OPTIMIZATION_WARMUP_STEPS:
                            relative_index_list = [relative_index for relative_index in multi_inputs.keys()]
                            if USE_DYNAMIC_MODELING_FLAG:
                                if DYNAMIC_TYPE=='mlp':
                                
                                    relative_box_residual = models.detector_residual(relative_index_list,world_outputs.embeddings)
                                    
                                elif DYNAMIC_TYPE=='vector_velocity':
                                    current_velocity = models['detector'].velocity #[1,8,3]
                                    current_velocity = current_velocity.unsqueeze(-2)
                                    relative_box_residual_list = [current_velocity *t for t in relative_index_list]
                                    relative_box_residual = torch.cat(relative_box_residual_list,dim=-2)
                                
                                elif DYNAMIC_TYPE=='scalar_velocity':
                                    current_velocity_scalar = models['detector'].scalar_velocity #[1,8,1]
                                    current_velocity_direction = world_outputs.velocity_direction #[1,8,3]
                                    current_velocity =  current_velocity_direction * current_velocity_scalar
                                    current_velocity = current_velocity.unsqueeze(-2)
                                    relative_box_residual_list = [current_velocity *t for t in relative_index_list]
                                    relative_box_residual = torch.cat(relative_box_residual_list,dim=-2)
                                
                                else:
                                    NotImplementedError
                            
                            else:
                                # vanallia VSRD
                                #FIXME Here
                                relative_box_residual = torch.zeros((1,nums_of_instance_number,nums_of_source_images_integrated_into_rendering,3)).to(device_id)


                        # ================================= multi-view projection ======================================================
                        multi_outputs = utils.DefaultDict(utils.Dict)
                                            
                        # This is the shared base 3D Bounding Boxes
                        world_boxes_3d = nn.functional.pad(world_outputs.boxes_3d, (0, 1), mode="constant", value=1.0) #(1,num_of_instances,8,4)


                        if USE_DYNAMIC_MODELING_FLAG:
                            if USE_DYNAMIC_MASK_FLAG:
                                # Align the dynamic mask with the current output.                            
                                dynamic_mask_for_target_view_for_output = get_dynamic_mask_for_the_world_output(target_inputs=target_inputs,
                                                                                                                world_boxes_3d=world_boxes_3d,
                                                                                                                dynamic_mask_for_target_view=dynamic_mask_for_target_view)

                                
                                
                        '''Get all box_3d(at each source view cam coordiante and its projected 2d boxes)'''
                        current_idx = 0
                        for relative_index, inputs in multi_inputs.items():
                            # Learn the Box Residual 
                            if step>=my_conf_train.TRAIN.OPTIMIZATION_WARMUP_STEPS:
                                # Get the current residual.
                                if USE_DYNAMIC_MODELING_FLAG:
                                    current_location_residual = relative_box_residual[:,:,current_idx,:] #(B,nums_of_instances,3)
                                else:
                                    # They are all Zeros for Vanallia VSRD
                                    current_batch_size = world_boxes_3d.shape[0]
                                    current_instance_numbers = world_boxes_3d.shape[1]
                                    current_location_residual = torch.zeros((current_batch_size,current_instance_numbers,3)).to(world_boxes_3d.device)
                                
                                # Using Dynamic Model Flg
                                if USE_DYNAMIC_MODELING_FLAG:
                                    if USE_DYNAMIC_MASK_FLAG:                                
                                        dynamic_mask_for_target_view_for_output_tensor = torch.from_numpy(np.array(dynamic_mask_for_target_view_for_output)).unsqueeze(0).unsqueeze(-1).to(current_location_residual.device).float()
                                        current_location_residual = current_location_residual * dynamic_mask_for_target_view_for_output_tensor
                                
                                current_world_boxes_3d = decode_box_3d(locations=world_outputs.locations,
                                                dimensions=world_outputs.dimensions,
                                                orientations=world_outputs.orientations,
                                                residual = current_location_residual)
                                current_world_boxes_3d = nn.functional.pad(current_world_boxes_3d, (0, 1), mode="constant", value=1.0) #(1,4,8,4)
                            
                            else:
                                # without the world boxes 3d
                                current_world_boxes_3d = world_boxes_3d
    
                                
                            # different idx for diferent instances
                            current_idx = current_idx + 1
                            # from world to camera coordinate at the target frames.
                            camera_boxes_3d = torch.einsum("bmn,b...n->b...m", inputs.extrinsic_matrices, current_world_boxes_3d)
                            camera_boxes_3d = camera_boxes_3d[..., :-1] / camera_boxes_3d[..., -1:] #(1,4,8,3)---> at source view camera coordinate

                            
                            # project the 3D bounding boxex from camera coordinate to image coordinate at current source vies
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
                                in zip(camera_boxes_3d, inputs.intrinsic_matrices)
                            ], dim=0) #(1,4,2,2)
                            
                            # make sure inside the image
                            camera_boxes_2d = torchvision.ops.clip_boxes_to_image(
                                boxes=camera_boxes_2d.flatten(-2, -1),
                                size=inputs.images.shape[-2:],
                            ).unflatten(-1, (2, 2))  #(1,4,2,2)
                            
                            # saved the camera_box_3d and camera_box_2d in all source views
                            multi_outputs[relative_index].update(
                                boxes_3d=camera_boxes_3d,
                                boxes_2d=camera_boxes_2d)



                        # get the current target outputs
                        target_outputs = multi_outputs[0]
                                            
                        # ----------------------------------------------------------------
                        # bipartite_matching
                        matching_cost_matrices = [
                            -torchvision.ops.distance_box_iou(
                                boxes1=pd_boxes_2d.flatten(-2, -1),
                                boxes2=gt_boxes_2d.flatten(-2, -1),
                            )
                            for pd_boxes_2d, gt_boxes_2d
                            in zip(target_outputs.boxes_2d, target_inputs.boxes_2d)
                        ]

                        matched_indices = list(map(
                            utils.torch_function(sp.optimize.linear_sum_assignment),
                            matching_cost_matrices,
                        )) # (tensor([0, 1, 2, 3]), tensor([1, 0, 2, 3]))]  第一个数组表示 target_outputs 的索引。第二个数组表示 target_inputs 的索引。


                        # ----------------------------------------------------------------
                        # projection loss
                        iou_projection_loss = torch.mean(torch.cat([
                            torch.cat([
                                torchvision.ops.distance_box_iou_loss(
                                    boxes1=pd_boxes_2d[pd_indices[visible_masks[gt_indices]], ...].flatten(-2, -1),
                                    boxes2=gt_boxes_2d[gt_indices[visible_masks[gt_indices]], ...].flatten(-2, -1),
                                    reduction="none",
                                )
                                for pd_boxes_2d, gt_boxes_2d, visible_masks, (pd_indices, gt_indices)
                                in zip(outputs.boxes_2d, inputs.boxes_2d, inputs.visible_masks, matched_indices)
                            ], dim=0)
                            for outputs, inputs in zip(multi_outputs.values(), multi_inputs.values())
                        ], dim=0))

                        l1_projection_loss = torch.mean(torch.cat([
                            torch.cat([
                                nn.functional.smooth_l1_loss(
                                    input=pd_boxes_2d[pd_indices[visible_masks[gt_indices]], ...].flatten(-2, -1),
                                    target=gt_boxes_2d[gt_indices[visible_masks[gt_indices]], ...].flatten(-2, -1),
                                    reduction="none",
                                )
                                for pd_boxes_2d, gt_boxes_2d, visible_masks, (pd_indices, gt_indices)
                                in zip(outputs.boxes_2d, inputs.boxes_2d, inputs.visible_masks, matched_indices)
                            ], dim=0)
                            for outputs, inputs in zip(multi_outputs.values(), multi_inputs.values())
                        ], dim=0))
                        
                        
                        
                        # ----------------------------------------------------------------
                        # instance loss
                        cosine_annealing = lambda x, a, b: (np.cos(np.pi * x) + 1.0) / 2.0 * (a - b) + b
                        cosine_ratio = step / my_conf_train.TRAIN.OPTIMIZATION_NUM_STEPS
                        
                        sdf_union_temperature = cosine_annealing(
                            step / my_conf_train.TRAIN.OPTIMIZATION_NUM_STEPS,
                            my_conf_train.TRAIN.VOLUME_RENDERING.MAX_SDF_UNION_TEMPERATURE,
                            my_conf_train.TRAIN.VOLUME_RENDERING.MIN_SDF_UNION_TEMPERATURE,
                        )
                        sdf_std_deviation = cosine_annealing(
                            step / my_conf_train.TRAIN.OPTIMIZATION_NUM_STEPS,
                            my_conf_train.TRAIN.VOLUME_RENDERING.MAX_SDF_STD_DEVIATION,
                            my_conf_train.TRAIN.VOLUME_RENDERING.MIN_SDF_STD_DEVIATION)


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

                        # bigger than optimized residual box steps: using RDF
                        if step >= my_conf_train.TRAIN.OPTIMIZATION_RESIDUAL_BOX_STEPS:
                            # Compute the Residual Signed Distance.
                            # #(1,4,1617), here the 1617 is the unit numbers.
                            distance_field_weights = models.hyper_distance_field(world_outputs.embeddings)             
                            world_outputs.update(distance_field_weights=distance_field_weights)

                            soft_distance_fields = []

                            for locations, dimensions, orientations, distance_field_weights in zip(
                                world_outputs.locations,
                                world_outputs.dimensions,
                                world_outputs.orientations,
                                world_outputs.distance_field_weights):
                                
                                
                                # 初始化一个列表来存储每个实例的距离场
                                distance_fields = []
                                # 遍历场景中的每个实例，提取位置、尺寸、方向和距离场权重，并生成实例标签
                                for instance_label, (location, dimension, orientation, distance_field_weights) in enumerate(zip(
                                    locations,
                                    dimensions,
                                    orientations,
                                    distance_field_weights)):
                                    
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


                                for current_frame_idx in range(len(multi_inputs.keys())):
                                    current_distance_fields = [d[current_frame_idx] for d in distance_fields] # get current distance fields
                                    # 将所有实例的距离场进行软联合
                                    soft_distance_field = soft_union(
                                        distance_fields=current_distance_fields,
                                        temperature=sdf_union_temperature)
                                    # 将处理后的软距离场添加到场景列表中: length should be eight
                                    soft_distance_fields.append(soft_distance_field)


                        # only learn the bouding boxes
                        else:
                            # learn the static
                            if step >= my_conf_train.TRAIN.OPTIMIZATION_WARMUP_STEPS:
                                # using dynamic modeling flag
                                if USE_DYNAMIC_MODELING_FLAG:
                                    boxes_residuals = relative_box_residual
                                else:
                                    boxes_residuals = torch.zeros(1,8,nums_of_source_images_integrated_into_rendering,3).to(device)
                            # learn the residual   
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


                        soft_distance_fields = [soft_distance_fields]
                        ''' Get the 1000 sample rays from the (multi camera positions) + (Multi direction)
                        '''
                        # 初始化一个列表来存储每个multi_soft_masks的最大值
                        max_values_list = []
                        # for each pixel, get the one most near to the instance
                        for multi_soft_mask in multi_soft_masks:
                            # for each location, get the biggest value from the potential N's instance soft mask
                            max_values = torch.max(multi_soft_mask, dim=-1).values # the last dimension is the instance
                            max_values_list.append(max_values) # here the length is only one

                        '''Sample Rays from the M View Images '''
                        # 将最大值列表堆叠成一个张量
                        stacked_max_values = torch.stack(max_values_list, dim=0) #Size([1,16, 376, 1408]):which is the value
                        
                        
                        # Tracking the rays, where the ray from
                        assert stacked_max_values.shape[1] == multi_camera_positions[0].shape[0]
                        sample_rays_nums_per_instance = divide_into_n_parts(my_conf_train.TRAIN.VOLUME_RENDERING.NUM_RAYS,stacked_max_values.shape[1])
                        assert stacked_max_values.shape[1] == len(sample_rays_nums_per_instance)
                        multi_ray_indices_list = []
                        for frame_index in range(stacked_max_values.shape[1]):
                            # how many rays sampled
                            sampled_rays_nums_current = sample_rays_nums_per_instance[frame_index]
                            current_stacked_max_values = stacked_max_values[:,frame_index,:,:]
                            current_flatten_max_values = current_stacked_max_values.flatten(1,-1)
                            current_multi_ray_indices = torch.multinomial(
                                input=current_flatten_max_values,
                                num_samples=sampled_rays_nums_current,
                                replacement=False
                            )
                            multi_ray_indices_list.append(current_multi_ray_indices)
                        
                        
                        # 8 multi rays
                        assert len(multi_ray_indices_list) == stacked_max_values.shape[1]                        
                        multi_sample_labels_list = []
                        multi_sample_gradients_list = []
                        
                        for idx in range(len(multi_ray_indices_list)):
                            multi_ray_indices_sub = [multi_ray_indices_list[idx]]
                            # Do the Rendering Here 
                            multi_sampled_labels, multi_sampled_gradients = map(torch.stack, zip(*[
                                hierarchical_wrapper(rendering.hierarchical_volumetric_rendering)(
                                    distance_field=soft_distance_field[idx],
                                    ray_positions=multi_camera_positions[idx:idx+1,:,:,:].flatten(0, -2)[multi_ray_indices, ...],
                                    ray_directions=multi_ray_directions[idx:idx+1,:,:,:].flatten(0, -2)[multi_ray_indices, ...],
                                    distance_range=my_conf_train.TRAIN.VOLUME_RENDERING.DISTANCE_RANGE,
                                    num_samples=my_conf_train.TRAIN.VOLUME_RENDERING.NUM_FINE_SAMPLES,
                                    sdf_std_deviation=sdf_std_deviation,
                                    cosine_ratio=cosine_ratio,
                                )
                                for (
                                    soft_distance_field,
                                    multi_camera_positions,
                                    multi_ray_directions,
                                    multi_ray_indices,
                                )
                                in zip(
                                    soft_distance_fields,
                                    multi_camera_positions,
                                    multi_ray_directions,
                                    multi_ray_indices_sub,
                                )
                            ]))
                            
                            multi_sample_labels_list.append(multi_sampled_labels)
                            multi_sample_gradients_list.append(multi_sampled_gradients)
                            
                                                
                        multi_sampled_labels = torch.cat(multi_sample_labels_list,dim=-2).squeeze(1)
                        multi_sampled_gradients = torch.cat(multi_sample_gradients_list,dim=-2).squeeze(-3)

                        # Get the Silhouette Loss Here 
                        silhouttle_loss_list_before_mean_list = []
                        for idx in range(len(multi_ray_indices_list)):
                            multi_ray_indices_sub = [multi_ray_indices_list[idx]]
                            current_cross_entropy = torch.stack([
                            nn.functional.binary_cross_entropy(
                                input=multi_sample_labels_list[idx][..., pd_indices].clamp(1.0e-6, 1.0 - 1.0e-6).squeeze(1),
                                target=multi_soft_masks[idx:idx+1,:,:,:].flatten(0, -2)[multi_ray_indices, ...][..., gt_indices],
                                reduction="none",
                            )
                            for (
                                multi_soft_masks,
                                multi_ray_indices,
                                (pd_indices, gt_indices),
                            )
                            in zip(
                                multi_soft_masks,
                                multi_ray_indices_sub,
                                matched_indices,
                            )
                            ], dim=0)
                            
                            silhouttle_loss_list_before_mean_list.append(current_cross_entropy)
                        
                        silhouttle_loss_list_before_mean = torch.cat(silhouttle_loss_list_before_mean_list,dim=-2).squeeze(1)
                        silhouette_loss = torch.mean(silhouttle_loss_list_before_mean)
                    



                        if USE_DYNAMIC_MODELING_FLAG:
                            if USE_DYNAMIC_MASK_FLAG:
                                losses = Dict(
                                    iou_projection_loss=iou_projection_loss,
                                    l1_projection_loss=l1_projection_loss,
                                    silhouette_loss=silhouette_loss)
                                
                        else:
                            losses = Dict(
                                iou_projection_loss=iou_projection_loss,
                                l1_projection_loss=l1_projection_loss,
                                silhouette_loss=silhouette_loss,
                            )
                            
                                

                        if step >= my_conf_train.TRAIN.OPTIMIZATION_WARMUP_STEPS:
                            eikonal_loss = nn.functional.mse_loss(
                                input=torch.norm(multi_sampled_gradients, dim=-1),
                                target=multi_sampled_gradients.new_ones(*multi_sampled_gradients.shape[:-1]),
                                reduction="mean",
                            )
                            losses.update(eikonal_loss=eikonal_loss)
                        
                        
                        loss = sum(loss * loss_weight_list[name] for name, loss in losses.items())
                        
                        meters.train.update(forward=stop_watch.restart())
                        torch.autograd.backward(loss)
                        meters.train.update(backward=stop_watch.restart())
                        optimizer.step()
                        scheduler.step()



                    # ----------------------------------------------------------------
                    # logging
                    with torch.no_grad():
                        if not (step + 1) % my_conf_train.TRAIN.LOGGING.SCALAR_INTERVALS:
                            # ----------------------------------------------------------------
                            # evaluation
                            pd_boxes_3d = [
                                pd_boxes_3d[pd_indices, ...] @ rectification_matrix.T
                                for pd_boxes_3d, rectification_matrix, (pd_indices, _)
                                in zip(target_outputs.boxes_3d, target_inputs.rectification_matrices, matched_indices)
                            ]
                            gt_boxes_3d = [
                                gt_boxes_3d[gt_indices, ...] @ rectification_matrix.T
                                for gt_boxes_3d, rectification_matrix, (_, gt_indices)
                                in zip(target_inputs.boxes_3d, target_inputs.rectification_matrices, matched_indices)
                            ]
                        
                            metrics = {}

                            if any([any(map(utils.compose(torch.isfinite, torch.all), gt_boxes_3d)) for gt_boxes_3d in gt_boxes_3d]):
                                rotation_matrix = rotation_matrix_x(torch.tensor(-np.pi / 2.0, device=device_id))
                                ious_3d, ious_bev = map(torch.as_tensor, zip(*sum([
                                    [
                                        box_3d_iou(
                                            corners1=pd_box_3d @ rotation_matrix.T,
                                            corners2=gt_box_3d @ rotation_matrix.T,
                                        )
                                        for pd_box_3d, gt_box_3d
                                        in zip(pd_boxes_3d, gt_boxes_3d)
                                        if torch.all(torch.isfinite(gt_box_3d))
                                    ]
                                    for pd_boxes_3d, gt_boxes_3d
                                    in zip(pd_boxes_3d, gt_boxes_3d)
                                ], [])))

                                iou_3d = torch.mean(ious_3d)
                                iou_bev = torch.mean(ious_bev)

                                accuracy_3d_25 = torch.mean((ious_3d > 0.25).float())
                                accuracy_bev_25 = torch.mean((ious_bev > 0.25).float())
                                accuracy_3d_50 = torch.mean((ious_3d > 0.50).float())
                                accuracy_bev_50 = torch.mean((ious_bev > 0.50).float())

                                metrics.update(
                                    iou_3d=iou_3d,
                                    iou_bev=iou_bev,
                                    accuracy_3d_25=accuracy_3d_25,
                                    accuracy_bev_25=accuracy_bev_25,
                                    accuracy_3d_50=accuracy_3d_50,
                                    accuracy_bev_50=accuracy_bev_50,
                                )

                            scalars = {
                                **{f"losses/{name}": loss.item() for name, loss in losses.items()},
                                **{f"metrics/{name}": metric.item() for name, metric in metrics.items()},
                                **{
                                    f"learning_rates/{param_group_name}": param_group["lr"]
                                    for param_group_name, param_group
                                    in zip(param_group_names, optimizer.param_groups)
                                },
                                **{
                                    f"hyperparameters/sdf_union_temperature": sdf_union_temperature,
                                    f"hyperparameters/sdf_std_deviation": sdf_std_deviation,
                                    f"hyperparameters/cosine_ratio": cosine_ratio,
                                },
                            }

                            logger.info(
                                f"[Training] Rank: {torch.distributed.get_rank()}, Step: {step}, Progress: {meters.train.progress():.2%}, "
                                f"ETA: {datetime.timedelta(seconds=meters.train.arrival_seconds())}, "
                                f"scalars: {json.dumps(scalars, indent=4)}"
                            )
                            logger.info(
                                f"[Training] Rank: {torch.distributed.get_rank()}, Step: {step}, Progress: {meters.train.progress():.2%}, "
                                f"ETA: {datetime.timedelta(seconds=meters.train.arrival_seconds())}, "
                                f"runtimes: {json.dumps(dict(zip(meters.train.keys(), meters.train.means())), indent=4)}"
                            )

                            for name, metric in scalars.items():
                                writer.add_scalar(f"scalars/{name}", metric, step)

                        if not (step + 1) % my_conf_train.TRAIN.LOGGING.CKPT_INTERVALS:
                            saver.save(
                                filename=f"step_{step}.pt",
                                step=step,
                                models={
                                    name: model.state_dict()
                                    for name, model in models.items()
                                },
                                optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(),
                                metrics=metrics,
                            )
                        meters.train.update(logging=stop_watch.restart())

        stop_watch.stop()
                            
                            
    train()
        


def parse_args():
    parser = argparse.ArgumentParser(description="SceneFlow-Multi-Baseline Images")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    parser.add_argument(
        "--device_id",
        type=int,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    # get the local rank
    args = parser.parse_args()


    return args

if __name__=="__main__":

    args = parse_args()
    main(args=args)
