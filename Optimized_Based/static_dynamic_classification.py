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
from preprocessing.Initial_Attributes.Get_Initial_Attributes import Get_Initial_Attributes
import argparse

from torch.utils.data import DataLoader

def main(args):

    
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
    
    loaders = DataLoader(dataset=datasets,
                         batch_size=1,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=False,
                         collate_fn=collate_nested_dicts)
    

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


        # dyanmic labels for targhet views
        dynamic_labels_for_target_view = dict()
        dynamic_labels_for_target_view["instance_ids"] = []
        dynamic_labels_for_target_view["dynamic_labels"] = []
    
    
    




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

    args = parser.parse_args()


    return args

if __name__=="__main__":
    
    args = parse_args()
    main(args=args)


