import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
sys.path.append("../..")
from preprocessing.Initial_Attributes.file_io_utils import read_pickle_file,save_to_pickle,visualize_point_cloud_with_axis,visualize_point_cloud_with_axis_with_two_box
from preprocessing.Initial_Attributes.Get_GT_Attribute import get_gt_location_velocity_orientation
from preprocessing.Initial_Attributes.geometry_op import encode_box_3d,rotation_matrix_y
from preprocessing.Initial_Attributes.post_processing import get_orientation_from_point_cloud
from preprocessing.Initial_Attributes.geometry_op import generate_point_cloud,rotation_matrix_y_single

from Optimized_Based.utils.box_geo import decode_box_3d




def Roi_LiDAR_Valid_Mask_Selection(ROI_LiDAR_list,threshold=120):
    instances_name_list = list(ROI_LiDAR_list.keys())
    valid_mask = [True] * len(instances_name_list)
    for idx, instance_name in enumerate(instances_name_list):
        instance_LiDAR_Rois = ROI_LiDAR_list[instance_name]    
        if instance_LiDAR_Rois is None:
            valid_mask[idx] = False
        else:
            if instance_LiDAR_Rois.shape[0]<threshold:
                valid_mask[idx] = False
    return valid_mask

def get_location_orientation(multi_inputs,dynamic_mask=False,threshold=120,device=None):
    
    for key in multi_inputs.keys():
        
        multi_inputs[key]["velo"] = multi_inputs[key]["velo"].unsqueeze(0)
        
        ROI_LiDAR_list = multi_inputs[key]['Roi_LiDAR_Dict']
        ROI_LIDAR_velocity = multi_inputs[key]["velo"]        
        target_instance_name_list = list(ROI_LiDAR_list.keys())
        
        # length is the N, N is the number of the instance
        valid_mask = Roi_LiDAR_Valid_Mask_Selection(ROI_LiDAR_list = ROI_LiDAR_list,threshold=threshold)
        estimation_location_list = []
        estimation_lidar_list = []

        # estimated velocity
        orientations_from_speed = nn.functional.normalize(ROI_LIDAR_velocity[..., [2, 0]], dim=-1) # x,z
        orientations_from_speed = rotation_matrix_y(*torch.unbind(orientations_from_speed, dim=-1))

        
        for idx in range(len(valid_mask)):
            # if valid mask
            if valid_mask[idx]:
                current_instance_lidar = ROI_LiDAR_list[target_instance_name_list[idx]]
            else:
                # using bi-direction-searching
                current_velocity = ROI_LIDAR_velocity[0][idx].unsqueeze(0)
                current_instance_idx = idx
                
    
                project_lidar, time_gap = bi_direction_searching(multi_inputs = multi_inputs,
                                       current_frame_idx = key,
                                       current_instance_idx = current_instance_idx,
                                       target_instance_name_list = target_instance_name_list,
                                       device=device)
                
                project_lidar = project_lidar.to(device)
                project_lidar = project_lidar + time_gap * current_velocity
                current_instance_lidar = project_lidar
                
            current_instance_lidar = current_instance_lidar.to(device)
            estimation_location = torch.mean(current_instance_lidar,dim=0,keepdim=True)
            
            estimation_lidar_list.append(current_instance_lidar)
            estimation_location_list.append(estimation_location)
            # multi_inputs[key]['est_lidars'] = estimation_lidar_list
            
            # dynamic             
            if dynamic_mask[idx]:
                orientations_from_speed = orientations_from_speed.float()
            
            # static
            else:
                if key ==0:
                    pass
                    # very slow speed: should be deleted
                    # est_orient = get_orientation_from_point_cloud(current_instance_lidar.cpu().numpy()) 
                    # cos = math.cos(est_orient)
                    # sin = math.sin(est_orient)
                    # orentation_matrix = rotation_matrix_y_single(cos=cos,sin=sin).unsqueeze(0).unsqueeze(0)
                    # orientations_from_speed[:,idx:idx+1,:,:] = orentation_matrix.to(orientations_from_speed.device)
                    # orientations_from_speed = orientations_from_speed.float()
            if key==0:
                multi_inputs[key]['est_orient'] = orientations_from_speed.float()
            
            

            
        multi_inputs[key]['est_loc'] = torch.cat(estimation_location_list,dim=0).unsqueeze(0)
    
    
    return multi_inputs
    
def bi_direction_searching(multi_inputs,current_frame_idx,current_instance_idx,target_instance_name_list,threshold=120,
                           device=None):
    
    nums_of_reference_images = len(list(multi_inputs.keys()))
    reference_image_list = sorted(list(multi_inputs.keys()))
    
    sorted_current_frame_idx = reference_image_list.index(current_frame_idx) # current
    forward_final_lidar = None
    backward_final_lidar = None
    
    time_gap = 0
    
    # forward searching : from i to max
    if sorted_current_frame_idx < nums_of_reference_images-1:
        for next_key in reference_image_list[sorted_current_frame_idx+1:]:
            Next_ROI_LiDAR_list = multi_inputs[next_key]['Roi_LiDAR_Dict']
            next_valid_mask = Roi_LiDAR_Valid_Mask_Selection(ROI_LiDAR_list = Next_ROI_LiDAR_list,threshold=threshold)
            
            if next_valid_mask[current_instance_idx]==True:
                forward_final_lidar = Next_ROI_LiDAR_list[target_instance_name_list[current_instance_idx]]
                time_gap = next_key - current_frame_idx
                break

    # backward searching : from i to 0 
    if sorted_current_frame_idx>0:
        for prev_key in reference_image_list[:sorted_current_frame_idx][::-1]:
            Prev_ROI_LiDAR_list = multi_inputs[prev_key]['Roi_LiDAR_Dict']
            prev_valid_mask = Roi_LiDAR_Valid_Mask_Selection(ROI_LiDAR_list = Prev_ROI_LiDAR_list,threshold=threshold)
        
            if prev_valid_mask[current_instance_idx]==True:
                backward_final_lidar = Prev_ROI_LiDAR_list[target_instance_name_list[current_instance_idx]]
                time_gap = prev_key - current_frame_idx
                break
            
    
    # Returns
    if backward_final_lidar is not None and forward_final_lidar is not None:
        backward_nums = backward_final_lidar.shape[0]
        forward_nums = forward_final_lidar.shape[0]
        if backward_nums>forward_nums:
            return backward_final_lidar,time_gap
        else:
            return forward_final_lidar,time_gap
        
    
    if backward_final_lidar is None and forward_final_lidar is not None:
        return forward_final_lidar,time_gap
    
    if backward_final_lidar is not None and forward_final_lidar is None:
        return backward_final_lidar,time_gap
    
    if backward_final_lidar is None and forward_final_lidar is None:
        return torch.randn((1,3)).to(device),time_gap
            
        

if __name__=="__main__":

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    input_example_path = "Debug_Examples/Example_With_RoI_LiDAR_Velocity.pkl"
    multi_inputs = read_pickle_file(input_example_path)
    
    dynamic_mask = [False,False,False,True,True,True]
    multi_inputs = get_location_orientation(multi_inputs=multi_inputs,dynamic_mask=dynamic_mask,threshold=120)
    

    locations,dimensions,orientations,gt_velocity = get_gt_location_velocity_orientation(multi_inputs=multi_inputs,
                                                                            dyanmic_mask_list=dynamic_mask,
                                                                             use_velocity_direction=True)
    
    
    
    
    align_bounding_box = decode_box_3d(locations=multi_inputs[0]['est_loc'].to(dimensions.device),
                                        dimensions=dimensions,
                                        orientations=multi_inputs[0]['est_orient'].to(dimensions.device))
    
    
    gt_bounding_box = decode_box_3d(locations=locations.to(dimensions.device),
                                        dimensions=dimensions,
                                        orientations=orientations.to(dimensions.device))
    
    
    visualize_point_cloud_with_axis_with_two_box(axis_vis=True,boxes_3d1=align_bounding_box[0].cpu().numpy(),boxes_3d2=gt_bounding_box[0].cpu().numpy())
    # print(multi_inputs[0]['est_loc'])
    # print(multi_inputs[0]['velo'])
    # print(multi_inputs[0]['est_orient'].shape)
    # print("---------------------------------------------")
    
    # print(locations)
    # print(gt_velocity)
    
