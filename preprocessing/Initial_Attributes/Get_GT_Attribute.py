import sys
sys.path.append("../..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from preprocessing.Initial_Attributes.file_io_utils import read_pickle_file,read_depth,get_depth_filename,save_to_pickle
from preprocessing.Initial_Attributes.geometry_op import transform_bounding_boxes_to_world,encode_box_3d,rotation_matrix_y
import numpy as np
from PIL import Image

def get_tensor_mean(tensor_list):
    mean_tensor = torch.zeros_like(tensor_list[0])
    for tensor in tensor_list:
        mean_tensor+=tensor
    mean_tensor = mean_tensor/len(tensor_list)
    return mean_tensor

def get_gt_velocity(data_inputs):
    
    ids = data_inputs[0]['instance_ids'][0].cpu().numpy().tolist()
    gt_speed_dict_accumulate = dict()
    gt_speed_dict_mean = dict()
    
    for id in ids:
        gt_speed_dict_accumulate[id] = []
        gt_speed_dict_mean[id] = []
        
    relative_frame_index = list(data_inputs.keys())
    sorted_relative_frame_index = sorted(relative_frame_index)
    
    
    for index in range(len(sorted_relative_frame_index)-1):
        
        current_frame_index = sorted_relative_frame_index[index]
        
        current_instances_ids = data_inputs[current_frame_index]['instance_ids']
        current_boxes3d = data_inputs[current_frame_index]['boxes_3d']
        current_extrinsic_matrices = data_inputs[current_frame_index]["extrinsic_matrices"]
        visible_current = data_inputs[current_frame_index]["visible_masks"][0]

        world3d_location_current = transform_bounding_boxes_to_world(extrinsic_matrix=torch.inverse(current_extrinsic_matrices),
                                                            bounding_boxes_camera=current_boxes3d[0])
        
        
        next_frame_index = sorted_relative_frame_index[index+1]
        next_instances_ids = data_inputs[next_frame_index]['instance_ids']
        next_boxes3d = data_inputs[next_frame_index]['boxes_3d']
        next_extrinsic_matrices = data_inputs[next_frame_index]["extrinsic_matrices"]
        visible_next = data_inputs[next_frame_index]["visible_masks"][0]


        world3d_location_next = transform_bounding_boxes_to_world(extrinsic_matrix=torch.inverse(next_extrinsic_matrices),
                                                            bounding_boxes_camera=next_boxes3d[0])
        
        
        assert visible_next.shape == visible_next.shape
        visible_all_mask = visible_next * visible_current
        time_gap = next_frame_index - current_frame_index
        
        velocity = world3d_location_next.mean(dim=1) - world3d_location_current.mean(dim=1)
        
        for sub_index in range(len(ids)):
            current_id = ids[sub_index]
            current_visible = visible_all_mask[sub_index]
            current_speed = velocity[sub_index]
                        
            if current_visible:
                gt_speed_dict_accumulate[current_id].append(current_speed/time_gap)
        
    for key in gt_speed_dict_accumulate:
        gt_speed_dict_mean[key] = get_tensor_mean(gt_speed_dict_accumulate[key])

    return gt_speed_dict_accumulate,gt_speed_dict_mean

def get_gt_location_velocity_orientation(multi_inputs,dyanmic_mask_list,use_velocity_direction=False):
    
    # make sure the dyanmic list is the same sequences
    target_inputs = multi_inputs[0]
    target_instance_list = multi_inputs[0]['instance_ids'][0].cpu().numpy().tolist()
    
    assert len(target_instance_list) == len(dyanmic_mask_list)


    boxes3d_at_relative_frame_id_0 = transform_bounding_boxes_to_world(extrinsic_matrix=torch.inverse(target_inputs['extrinsic_matrices']),
                                                                        bounding_boxes_camera=target_inputs['boxes_3d'][0])
    # [1,6,3] , [1,6,3] , [1,6,3,3]
    locations, dimensions, orientations = encode_box_3d(boxes3d_at_relative_frame_id_0.unsqueeze(0))

    gt_velocity_dict_accumulate,gt_velocity_mean = get_gt_velocity(data_inputs=multi_inputs)
    my_velocity_list = []
    for key in gt_velocity_dict_accumulate.keys():
        my_velocity_list.append(gt_velocity_mean[key].unsqueeze(0))
    gt_velocity = torch.cat(my_velocity_list,dim=0).unsqueeze(0)
    
    if use_velocity_direction:
        orientations_from_speed = nn.functional.normalize(gt_velocity[..., [2, 0]], dim=-1) # x,z
        orientations_from_speed = rotation_matrix_y(*torch.unbind(orientations_from_speed, dim=-1))
        
        # dynamic mask
        condition = torch.tensor(dyanmic_mask_list, dtype=torch.bool).view(1, -1, 1, 1).to(gt_velocity.device)
        orientations = torch.where(condition, orientations_from_speed, orientations)
        
    
    return locations,dimensions,orientations,gt_velocity
    
    
if __name__=="__main__":

    ''' Input Example and Dynamic List '''
    input_example_path = "Debug_Examples/input_example_with_depth.pkl"
    multi_inputs = read_pickle_file(input_example_path)
    dynamic_list_example = [False,False,False,True,True,True]
    
    '''Logic Begin Here'''
    locations,dimensions,orientations,gt_velocity = get_gt_location_velocity_orientation(multi_inputs=multi_inputs,dyanmic_mask_list=dynamic_list_example,
                                                                             use_velocity_direction=True)

    


    
    

    
    

    
    
    
    
