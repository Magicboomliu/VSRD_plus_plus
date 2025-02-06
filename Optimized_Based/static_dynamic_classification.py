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
from preprocessing.Initial_Attributes.Test_Velocity_API import Get_Estimated_Velocity
import argparse

from torch.utils.data import DataLoader


def before_with_2013_string(string):
    return string[:string.index("2013")]


def biltaral_matching(target_string_lst,source_string_lst):

    shared_string_for_target = []
    target_string_index = []
    source_string_index = []
       
    for idx, target_string in enumerate(target_string_lst):
        if target_string in source_string_lst:
            shared_string_for_target.append(target_string)
            target_string_index.append(idx)
            source_string_index.append(source_string_lst.index(target_string))
    
    return shared_string_for_target,target_string_index,source_string_index


def load_pickle(path):
    with open(path, "rb") as f:
        loaded_data = pickle.load(f)
    
    return loaded_data



import pickle
def SAVE_INTO_PICKLE(dict,path):
    # 将字典保存为 YAML 文件
    with open(path, "wb") as f:
        pickle.dump(dict, f)


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
        

    saved_dict_path = os.path.join("estimated_dynamic_{}.pkl".format(args.config_path))

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
    velocity_dict = dict()
    

    for multi_inputs in vsrd.distributed.tqdm(loaders):        
        try:
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


            # Get the ground truth instance_ids and the dynamic labels
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
                

            dynamic_mask_for_target_view = dynamic_labels_for_target_view['dynamic_labels']
            dynamic_mask_for_target_view = [bool(int(float(data))) for data in dynamic_mask_for_target_view.split(",")]
            multi_inputs = Get_Estimated_Velocity(multi_inputs=multi_inputs,
                                                dynamic_mask_list=None,
                                                device='cuda:0')
            
            est_velocity = multi_inputs[0]['velo'].float().contiguous().to(device_id)
            multi_views_inputs_instance_ids = multi_inputs[0]['instance_ids'][0].tolist()
            dynamic_gt_instance_list = [int(s) for s in dynamic_labels_for_target_view["instance_ids"].split(",")]
        
            
            shared_string_for_target,target_string_index,source_string_index =biltaral_matching(target_string_lst=multi_views_inputs_instance_ids,
                            source_string_lst=dynamic_gt_instance_list)

            # here the veloicty should be the [N,3] tensor
            
            est_velocity = est_velocity[target_string_index,:].cpu().numpy()
            dynamic_labels_for_target_view = [dynamic_mask_for_target_view[i] for i in source_string_index]
            current_fname = multi_inputs[0]['filenames'][0]

            velocity_dict[current_fname] = dict()
            velocity_dict[current_fname]['est_velo'] = est_velocity
            velocity_dict[current_fname]['dynamic_gt'] = dynamic_labels_for_target_view
            velocity_dict[current_fname]['instance_name'] = shared_string_for_target

        except:
            pass
        

        

    SAVE_INTO_PICKLE(dict=velocity_dict,
                     path=saved_dict_path)
    
    

    

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


