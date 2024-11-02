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

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


# rendering
from vsrd.rendering.utils import ray_casting
from vsrd import rendering

# projection
from vsrd.operations.geometric_operations import project_box_3d
from vsrd.operations.geometric_operations import rotation_matrix,rotation_matrix_x,rotation_matrix_y,rotation_matrix_z
from vsrd.operations.kitti360_operations import box_3d_iou
from vsrd import visualization
from vsrd.utils import Dict

import logging
from tqdm import tqdm
import skimage.io
import json
from configs import conf_val

from Optimized_Based.utils.box_geo import decode_box_3d,divide_into_n_parts,get_dynamic_mask_for_the_world_output
from Optimized_Based.utils.file_io import read_text_lines,read_complex_strings
import re


LINE_INDICES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def format_int_to_string(number):
    # 将整数转换为长度为10的字符串，左侧填充零
    formatted_number = f"{number:010d}"
    return formatted_number

# save to json files
def save_to_json_files(data,filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        

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



def Inference():
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
    # get the datasets
    current_filanems = conf_val.TRAIN.DATASET.FILENAMES
    class_names = conf_val.TRAIN.DATASET.CLASS_NAMES
    num_of_workers = conf_val.TRAIN.DATASET.NUMS_OF_WORKERS
    num_source_frames = conf_val.TRAIN.DATASET.NUM_SOURCE_FRAMES
    
    target_transforms_resize_size = conf_val.TRAIN.DATASET.TARGET_TRANSFORMS.IMAGE_SIZE 
    target_transforms_min_mask_area1 = conf_val.TRAIN.DATASET.TARGET_TRANSFORMS.MIN_MASK_AREA_01
    target_transforms_min_mask_area2 = conf_val.TRAIN.DATASET.TARGET_TRANSFORMS.MIN_MASK_AREA_02
    target_transforms_min_box_size = conf_val.TRAIN.DATASET.TARGET_TRANSFORMS.MIN_BOX_SIZE
    
    source_transforms_resize_size = conf_val.TRAIN.DATASET.SOURCE_TRANSFORMS.IMAGE_SIZE 
    source_transforms_min_mask_area1 = conf_val.TRAIN.DATASET.SOURCE_TRANSFORMS.MIN_MASK_AREA_01
    source_transforms_min_mask_area2 = conf_val.TRAIN.DATASET.SOURCE_TRANSFORMS.MIN_MASK_AREA_02
    source_transforms_min_box_size = conf_val.TRAIN.DATASET.SOURCE_TRANSFORMS.MIN_BOX_SIZE
    dataset_rectification = conf_val.TRAIN.DATASET.RECTIFICATION    
    train_batch_size = conf_val.TRAIN.DATASET.BATCH_SIZE
    
    
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
    datasets = KITTI360Dataset(filenames=current_filanems,
                              class_names=class_names,
                              num_of_workers=num_of_workers,
                              num_source_frames=num_source_frames,
                              target_transforms=target_transforms,
                              source_transforms=source_transforms,
                              rectification=dataset_rectification)

    # loaders
    loaders = DataLoader(datasets, batch_size=train_batch_size, collate_fn=collate_nested_dicts)
    #======================================================================================================
    for multi_inputs in tqdm(loaders): 
        
        multi_inputs = {
            relative_index: Dict.apply({
                key if re.fullmatch(r".*_\dd", key) else inflection.pluralize(key): value
                for key, value in inputs.items()
            })
            for relative_index, inputs in multi_inputs.items()
        }
        
        
        multi_inputs = utils.to(multi_inputs, device=0, non_blocking=True)
        target_inputs = multi_inputs[0]
        num_instances, = map(len, target_inputs.hard_masks)


        image_filename, = target_inputs.filenames    #/media/zliu/data12/dataset/KITTI/VSRD_Format/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000793.png
        root_dirname = datasets.get_root_dirname(image_filename)  #/media/zliu/data12/dataset/KITTI/VSRD_Format
        image_dirname = os.path.splitext(os.path.relpath(image_filename, root_dirname))[0] #data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000793
        
        logger = utils.get_logger(image_dirname)

        # Using Dynamic Masks Or Not
        USE_DYNAMIC_MASK_FLAG = conf_val.TRAIN.USE_DYNAMIC_MASK
        DYNAMIC_TYPE = conf_val.TRAIN.DYNAMIC_MODELING_TYPE
        # Using Dynamic Mask Modeling
        USE_DYNAMIC_MODELING_FLAG = conf_val.TRAIN.USE_DYNAMIC_MODELING
        # Using RDF or Not
        USE_RDF_MODELING_FLAG = conf_val.TRAIN.USE_RDF_MODELING


        # Loading the Models 
        ckpt = torch.load(conf_val.EVAL.MODELS_PATH)
        models_weights = ckpt['models'] # keys are dict_keys(['detector', 'hyper_distance_field', 'positional_encoder'])

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
            model.to(0)


        # Loaded the Weights from the Per-trained Modelss
        for model_name, model in models.items():
            models[model_name].load_state_dict(models_weights[model_name])
        print("Loaded all the pre-trained Models")


        

        ''' Instance Alignment'''
        for source_inputs in multi_inputs.values():
            # each every time it will update, so jsyt fine.
            source_instance_indices = [] #[tensor([0, 1, 2, 3], device='cuda:0')]
            # 遍历 source_inputs 和 target_inputs 中的 instance_ids
            for source_instance_ids, target_instance_ids in zip(source_inputs.instance_ids, target_inputs.instance_ids):
                # 将 target_instance_ids 映射到 source_instance_ids 中的索引
                indices = [
                    source_instance_ids.tolist().index(target_instance_id.item()) 
                    if target_instance_id in source_instance_ids else -1 
                    for target_instance_id in target_instance_ids
                ]  # [-1, 4, 5, 6, 7, 8, 11, 17]
                # 将索引列表转换为张量并添加到 source_instance_indices
                source_instance_indices.append(source_instance_ids.new_tensor(indices)) # length should be
            
    
            # instance by instance 
            source_labels = [
                utils.reversed_pad(source_labels, (0, 1))[source_instance_indices, ...]
                for source_labels, source_instance_indices
                in zip(source_inputs.labels, source_instance_indices)
            ] #[tensor([0, 0, 0, 0], device='cuda:0')]
            
            
            
            source_boxes_2d = [
                utils.reversed_pad(source_boxes_2d, (0, 1))[source_instance_indices, ...]
                for source_boxes_2d, source_instance_indices
                in zip(source_inputs.boxes_2d, source_instance_indices)
            ]

            
            source_boxes_3d = [
                utils.reversed_pad(source_boxes_3d, (0, 1))[source_instance_indices, ...]
                for source_boxes_3d, source_instance_indices
                in zip(source_inputs.boxes_3d, source_instance_indices)
            ] # list [[boxed_3d]], where the boxed 3d is [N,8,3]
            

            
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
            ]

            source_inputs.update(
                labels=source_labels,
                boxes_2d=source_boxes_2d,
                boxes_3d=source_boxes_3d,
                hard_masks=source_hard_masks,
                soft_masks=source_soft_masks,
                instance_ids=source_instance_ids,
                visible_masks=source_visible_masks,
            )
            


        # recodered the instance ids for the final results.
        source_instance_ids_for_all_reference = multi_inputs[0].instance_ids[0]
        
        nums_of_source_images_integrated_into_rendering = len(multi_inputs.keys())
        nums_of_instance_number = len(source_instance_ids_for_all_reference)

        # get the static and dyanmic mask uisng proposed filetring algorithim.
        dynamic_labels_for_target_view = dict()
        dynamic_labels_for_target_view["instance_ids"] = []
        dynamic_labels_for_target_view["dynamic_labels"] = []
        if USE_DYNAMIC_MODELING_FLAG:
            if USE_DYNAMIC_MASK_FLAG:
                dynamic_raw_contents = read_text_lines(conf_val.TRAIN.DYNAMIC_LABELS_PATH)
                for content in dynamic_raw_contents:
                    content = content.strip()
                    current_returned_dict = read_complex_strings(content)
                    if current_returned_dict['filename'] == image_filename:
                        dynamic_labels_for_target_view['instance_ids'] = current_returned_dict['instance_ids']
                        dynamic_labels_for_target_view["dynamic_labels"] = current_returned_dict['labels']
                
                dynamic_mask_for_target_view = dynamic_labels_for_target_view['dynamic_labels']
                dynamic_mask_for_target_view = [bool(int(float(data))) for data in dynamic_mask_for_target_view.split(",")]



        # Doing the Inference Here
        with torch.no_grad():
            world_outputs = utils.Dict.apply(models.detector()) #['boxes_3d', 'locations', 'dimensions', 'orientations', 'embeddings']
            multi_outputs = utils.DefaultDict(utils.Dict)
            world_boxes_3d = nn.functional.pad(world_outputs.boxes_3d, (0, 1), mode="constant", value=1.0) #(1,4,8,4)
            
            relative_index_list = [relative_index for relative_index in multi_inputs.keys()]
        
            # Using Dynamic Mask or Not: Instance-Matching
            if USE_DYNAMIC_MODELING_FLAG:
                if USE_DYNAMIC_MASK_FLAG:
                    # Align the dynamic mask with the current output.                            
                    dynamic_mask_for_target_view_for_output = get_dynamic_mask_for_the_world_output(target_inputs=target_inputs,
                                                                                                    world_boxes_3d=world_boxes_3d,
                                                                                                    dynamic_mask_for_target_view=dynamic_mask_for_target_view)

                
            # Get the Time-Wise Residual
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
                    
                    # print(relative_box_residual)
                else:
                    raise NotImplementedError
            else:
                # vanallia VSRD
                #FIXME Here
                relative_box_residual = torch.zeros((1,nums_of_instance_number,nums_of_source_images_integrated_into_rendering,3)).to("cuda:0")       
            
            
            
            current_idx = 0
            '''Get all box_3d(at each source view cam coordiante and its projected 2d boxes)'''
            for relative_index, inputs in multi_inputs.items():
                # from world to camera coordinate.
                current_location_residual = relative_box_residual[:,:,current_idx,:]

                if USE_DYNAMIC_MASK_FLAG:
                    # Static Mask
                    dynamic_mask_for_target_view_for_output_tensor = torch.from_numpy(np.array(dynamic_mask_for_target_view_for_output)).unsqueeze(0).unsqueeze(-1).to(current_location_residual.device).float()
                    current_location_residual = current_location_residual * dynamic_mask_for_target_view_for_output_tensor

                current_world_boxes_3d = decode_box_3d(locations=world_outputs.locations,
                                dimensions=world_outputs.dimensions,
                                orientations=world_outputs.orientations,
                                residual= current_location_residual)
                
                current_world_boxes_3d = nn.functional.pad(current_world_boxes_3d, (0, 1), mode="constant", value=1.0) #(1,4,8,4)
                current_idx = current_idx + 1
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
                    boxes_2d=camera_boxes_2d,
                )


            # create saved jsons            
            saved_evaluation_path = image_filename[len(root_dirname)+1:-len(os.path.basename(image_filename))]
            saved_evaluation_path = os.path.join("evaluation_results/{}".format(conf_val.TRAIN.MODEL_TYPE),saved_evaluation_path)
            os.makedirs(saved_evaluation_path,exist_ok=True)
            
        
            # Evaluation All the Frames 
            for frame_index,(relative_index, inputs) in enumerate(multi_inputs.items()):
                current_saved_name_mean = format_int_to_string(int(os.path.basename(image_filename)[:-4]) + relative_index) + ".json"
                os.makedirs(os.path.join(saved_evaluation_path,"mean"),exist_ok=True)
                current_saved_name_mean = os.path.join(os.path.join(saved_evaluation_path,"mean"),current_saved_name_mean)

                current_saved_name_specific = format_int_to_string(int(os.path.basename(image_filename)[:-4]) + relative_index) + ".json"
                os.makedirs(os.path.join(saved_evaluation_path,"specific"),exist_ok=True)
                current_saved_name_specific = os.path.join(os.path.join(saved_evaluation_path,'specific'),current_saved_name_specific)
                
                # The Muli inputs and the Multi Outputs All have M, where M is the multi-frame numbsers
                # get the current target outputs
                target_outputs = multi_outputs[relative_index]
                target_inputs = multi_inputs[relative_index] # instance ids: I want to get the instance IDs
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
                
                matching_cost_matrices = [torch.nan_to_num(matrix, nan=-1000000) for matrix in matching_cost_matrices]

                
                matched_indices = list(map(
                    utils.torch_function(sp.optimize.linear_sum_assignment),
                    matching_cost_matrices,
                )) # (tensor([0, 1, 2, 3]), tensor([1, 0, 2, 3]))]  第一个数组表示 target_outputs 的索引。第二个数组表示 target_inputs 的索引。

            
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
                
                instance_id_list = [
                    instance_ids[gt_indices, ...]
                    for instance_ids,(_, gt_indices)
                    in zip(target_inputs.instance_ids,  matched_indices)
                ]
                
                visible_masks_list = [
                    visible_masks[gt_indices, ...]
                    for visible_masks,(_, gt_indices)
                    in zip(target_inputs.visible_masks,  matched_indices)
                ]
                
                
                metrics = {}
                if any([any(map(utils.compose(torch.isfinite, torch.all), gt_boxes_3d)) for gt_boxes_3d in gt_boxes_3d]):
                    rotation_matrix = rotation_matrix_x(torch.tensor(-np.pi / 2.0, device=0))
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
                    
                    
             
                    # save the information for this instance
                    idx = 0
                    each_instance_metrics = {}
                    for instance_id_name in instance_id_list[0]:
                        
                        # # if the instance in the target view is visible, save into jsons.
                        if visible_masks_list[0][idx].data.item():
                            instance_metric  = {}
                            iou_3d_per_instance = ious_3d[idx].data.item()
                            iou_bev_per_instance = ious_bev[idx].data.item()
                            instance_metric['iou_3d'] = iou_3d_per_instance
                            instance_metric['iou_bev'] = iou_bev_per_instance                      
                            each_instance_metrics[instance_id_name.data.item()] = instance_metric
                        idx = idx + 1
                        
                        
                    # instance-wise metric
                    save_to_json_files(each_instance_metrics,current_saved_name_specific)
                    

                   

                    iou_3d = torch.mean(ious_3d)
                    iou_bev = torch.mean(ious_bev)

                    accuracy_3d_25 = torch.mean((ious_3d > 0.25).float())
                    accuracy_bev_25 = torch.mean((ious_bev > 0.25).float())

                    accuracy_3d_50 = torch.mean((ious_3d > 0.50).float())
                    accuracy_bev_50 = torch.mean((ious_bev > 0.50).float())

                    metrics.update(
                        iou_3d=float(iou_3d.cpu().numpy()),
                        iou_bev=float(iou_bev.cpu().numpy()),
                        accuracy_3d_25=float(accuracy_3d_25.cpu().numpy()),
                        accuracy_bev_25 = float(accuracy_bev_25.cpu().numpy()),
                        accuracy_3d_50 = float(accuracy_3d_50.cpu().numpy()),
                        accuracy_bev_50 =float(accuracy_bev_50.cpu().numpy()),
                    )
                    
                    # overall metric
                    save_to_json_files(metrics,current_saved_name_mean)
        



if __name__=="__main__":
    Inference()