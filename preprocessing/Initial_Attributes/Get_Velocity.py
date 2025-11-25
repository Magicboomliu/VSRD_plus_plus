import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import pickle
sys.path.append("../..")
from PIL import Image
from preprocessing.Initial_Attributes.file_io_utils import read_pickle_file
from preprocessing.Initial_Attributes.Get_GT_Attribute import get_gt_location_velocity_orientation
from preprocessing.Initial_Attributes.geometry_op import remove_largest_and_smallest_point_clouds,remove_point_cloud_with_largest_avg_z,Sort_valid_ROI_LiDAR_by_relative_id
from preprocessing.Initial_Attributes.post_processing import icp_translation_only
from preprocessing.Initial_Attributes.file_io_utils import visualize_point_cloud_with_axis,save_to_pickle


def remove_max_min_z(tensor_list):
    if len(tensor_list) == 0:
        # 如果 tensor_list 为空，返回空的 tensor_list 和空的 indices_to_remove
        return tensor_list, []

    # 提取每个 tensor 的 z 值（即每个 tensor 的最后一个元素）
    z_values = torch.tensor([t[0, 2].item() for t in tensor_list])

    # 如果 z_values 为空，提前返回
    if z_values.numel() == 0:
        return tensor_list, []

    # 计算 z 值的绝对值
    z_abs = torch.abs(z_values)

    # 如果 z_abs 为空或只有一个元素，提前返回，避免 argmax/argmin 出错
    if z_abs.numel() <= 1:
        return tensor_list, []

    # 找到 z 值绝对值最大和最小的下标
    max_idx = torch.argmax(z_abs).item()  # 绝对值最大的下标
    min_idx = torch.argmin(z_abs).item()  # 绝对值最小的下标

    # 返回删除前的 z 值的下标
    indices_to_remove = [max_idx, min_idx]

    # 删除下标对应的元素
    tensor_list = [tensor for i, tensor in enumerate(tensor_list) if i not in indices_to_remove]
    
    return tensor_list, indices_to_remove

def get_velocity_from_multi_inputs(instance_accumulate_tuple_list,multi_inputs,static=False,device=None):

    # speed by the frame ids
    frame_idx_list = [a[1] for a in instance_accumulate_tuple_list]
    pc_list = [a[0] for a in instance_accumulate_tuple_list]
    
    total_nums = 0
    final_velcoity = 0
    velocity_list = []
    instance_lidar_nums = []
    
    # This is the instance by instance
    for ind  in range(len(frame_idx_list)-1):
        
        frame_idx = frame_idx_list[ind]
        next_frame_idx = frame_idx_list[ind+1]
        
        time_gap = next_frame_idx - frame_idx
        
        pc_squeezed = pc_list[ind]
        next_pc_squeezed = pc_list[ind+1]
        
        translation= icp_translation_only(A=pc_squeezed.cpu().numpy(),B=next_pc_squeezed.cpu().numpy())
        translation = torch.from_numpy(translation).to(pc_squeezed.device)
        velocity = translation
        
        if time_gap<0:
            continue
        elif time_gap>15:
            continue
        else:
            velocity = velocity*1.0 /time_gap
            velocity_list.append(velocity)
            nums_of_instance = pc_squeezed.shape[0]
            instance_lidar_nums.append(nums_of_instance)
    
    tensor_list, indices_to_remove = remove_max_min_z(tensor_list=velocity_list)
    
    for ind in range(len(velocity_list)):
        if ind not in indices_to_remove:
            velocity = velocity_list[ind]
            nums_of_instance = instance_lidar_nums[ind]

            final_velcoity +=velocity * nums_of_instance
            total_nums = total_nums + nums_of_instance
    
    if total_nums ==0:
        final_velcoity = torch.zeros((1,3)).to(device)
    else:
        final_velcoity = final_velcoity / total_nums
    

    # Static Modeling
    if static:
        if torch.max(final_velcoity)<0.5:
            final_velcoity = final_velcoity * 0.1
    else:
        # X
        if final_velcoity[0][0]>0.02:
            final_velcoity[0][0] = torch.tensor(0.02).type_as(final_velcoity) + 0.01 * final_velcoity[0][0]
        
        # Y
        if final_velcoity[0][1]>0.02:
            final_velcoity[0][1] = torch.tensor(0.02).type_as(final_velcoity) + 0.01 * final_velcoity[0][1]
        


    return final_velcoity
                
def Get_Estimated_Velocity_Using_CPI(multi_inputs,dynamic_mask=None,device=None):

    '''Get Estimated Velocity'''
    target_instance_ids = multi_inputs[0]['instance_ids'][0].cpu().numpy().tolist()
    
    # Accumulated Valid Velocity
    accmulate_valid_speed_dict = dict()
    for target_instance_id in target_instance_ids:
        accmulate_valid_speed_dict[target_instance_id] = []

    for key in multi_inputs.keys():
        ROI_LiDAR_Dict = multi_inputs[key]["Roi_LiDAR_Dict"]
        projected_mask = multi_inputs[key]["projected_valid_mask"]
        
        for instance_name in ROI_LiDAR_Dict.keys():
            if ROI_LiDAR_Dict[instance_name] is not None:
                if ROI_LiDAR_Dict[instance_name].shape[0]>100:
                    after_clustering_LiDAR = ROI_LiDAR_Dict[instance_name]
                    if after_clustering_LiDAR is not None:
                        saved_lidar_item = (after_clustering_LiDAR,int(key))
                        accmulate_valid_speed_dict[instance_name].append(saved_lidar_item)
    
    # Instance Mask
    estimated_velocity_list = []
    for idx, instance_name in enumerate(target_instance_ids):
        if dynamic_mask is not None:
            # static object
            if dynamic_mask[idx]:
                estimated_velocity = get_velocity_from_multi_inputs(instance_accumulate_tuple_list=accmulate_valid_speed_dict[instance_name],
                                                                    multi_inputs=multi_inputs,static=False,
                                                                    device=device)
            else:
                estimated_velocity = get_velocity_from_multi_inputs(instance_accumulate_tuple_list=accmulate_valid_speed_dict[instance_name],
                                                                    multi_inputs=multi_inputs,static=True,
                                                                    device=device)
        else:
            estimated_velocity = get_velocity_from_multi_inputs(instance_accumulate_tuple_list=accmulate_valid_speed_dict[instance_name],
                                                                multi_inputs=multi_inputs,static=True,
                                                                device=device)
        estimated_velocity_list.append(estimated_velocity)
    
    estimated_velocity_list = [tensor.to(device) for tensor in estimated_velocity_list]
    estimated_velocity = torch.cat(estimated_velocity_list,dim=0)
    

    return estimated_velocity
    
    

if __name__=="__main__":

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    input_example_path = "Debug_Examples/example_with_ROI_LIDAR.pkl"
    multi_inputs = read_pickle_file(input_example_path)
    

    '''Get the Location/Dimension/Orientations'''
    dynamic_list_example = [False,False,False,True,True,True]
    locations,dimensions,orientations,gt_velocity = get_gt_location_velocity_orientation(multi_inputs=multi_inputs,
                                                                            dyanmic_mask_list=dynamic_list_example,
                                                                             use_velocity_direction=True)
    
    '''Get the Estimated Velocity'''
    estimated_velocity = Get_Estimated_Velocity_Using_CPI(multi_inputs=multi_inputs,dynamic_mask=dynamic_list_example)
    
    
    ''' Debugger'''
    # Update Multi_Inputs
    for key in multi_inputs.keys():
        multi_inputs[key]["velo"] = estimated_velocity
    
    save_to_pickle("Debug_Examples/Example_With_RoI_LiDAR_Velocity.pkl",saved_dict=multi_inputs)
    
    
    
    
    
    



