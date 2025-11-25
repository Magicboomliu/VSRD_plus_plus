import sys
sys.path.append("../..")
import os
from preprocessing.Initial_Attributes.file_io_utils import read_pickle_file,read_depth,get_depth_filename,save_to_pickle,read_image_tensor,merge_for_visualization_version_1,visualize_point_cloud_with_axis
from preprocessing.Initial_Attributes.geometry_op import generate_point_cloud,match_instance_masks_with_iou,Convert_Disparity_to_Depth,disp_warp,transform_bounding_boxes_to_world,point_cloud_to_2d_mask
from preprocessing.Initial_Attributes.post_processing import update_instance_masks,compute_point_density,Obtain_Cluster_Single_PC
from preprocessing.dyanmic_static_filtering.preprocess import warp_image_by_flow,warp_image_by_depth,merge_mask_into_whole,shrink_masks_torch
import torch
import numpy as np
from PIL import Image

def Get_Pseudo_Roi_LiDAR(data_inputs,target_frame_instance_ids,
                         mask_shrink_threshold=0.25,
                         warpped_error_mask_threshold=0.40,
                         pc_density_radius=0.5,
                         min_density_ratio=0.01,
                         threeD_visualization=False):
    '''
    return a dict, 
    instance ids: roi point cloud
    if not visible: return None
    '''
    ROI_LiDAR_Dict = dict()
    filename = data_inputs['filenames'][0]
    instance_ids = data_inputs['instance_ids'][0].cpu().numpy().tolist()
    image = data_inputs['images']
    boxes_2d = data_inputs['boxes_2d']
    boxes_3d = data_inputs['boxes_3d']

    
    
    masks = data_inputs['masks'] # conduct the matching here
    extrinsic_matrices = data_inputs["extrinsic_matrices"]
    intrinsic_matrices = data_inputs["intrinsic_matrices"]
    visible_masks = data_inputs["visible_masks"][0].cpu().numpy().tolist()
    depth = data_inputs['pseudo_depth']
    
    masks = match_instance_masks_with_iou(instance_masks=masks[0],bounding_boxes=boxes_2d[0])
    masks = masks.unsqueeze(0) #[1,N,H,W]
    
    # Step 1: Shrink Morologic Operation
    masks[0]= shrink_masks_torch(instance_masks=masks[0],shrink_ratio=mask_shrink_threshold)
    
    # Step 2: Min Warping Error Selection
    left_name = filename
    right_name = left_name.replace("image_00","image_01")
    left_image_data = read_image_tensor(left_name).to(image.device)
    right_image_data = read_image_tensor(right_name).to(image.device)
    left_disparity = Convert_Disparity_to_Depth(disparity=depth) #[1,1,H,W]
    warped_left_data,valid_mask_left = disp_warp(img=right_image_data,disp=left_disparity)
    warped_left_data = warped_left_data * valid_mask_left
    warped_error = torch.sum(torch.abs(warped_left_data-left_image_data),dim=1,keepdim=True) * valid_mask_left[:,0:1,:,:] #[1,1,H,W]
    for sub_ind in range(masks[0].shape[0]):
        masks[0][sub_ind] = valid_mask_left[:,0,:,:].squeeze(0) * masks[0][sub_ind]
    
    masks= update_instance_masks(warp_error=warped_error,instance_masks=masks,ratio=warpped_error_mask_threshold)
    # target frame bounding boxes in the world
    boxes_3d_target_frame = transform_bounding_boxes_to_world(extrinsic_matrix=torch.inverse(extrinsic_matrices),
                                                              bounding_boxes_camera=boxes_3d[0])

    # get the point cloud
    pc = generate_point_cloud(image=image,
                         intrinsics=intrinsic_matrices,
                         extrinsics=extrinsic_matrices,
                         depth_map=depth) #[1,3,H,W]
    

    # Step3: Density-based Filtering
    instance_total_nums = masks[0].shape[0]
    for instance_ind in range(instance_total_nums):
        instance_mask_bool = masks[0][instance_ind].bool().repeat(1,3,1,1)
        instance_pc_filtered = pc[instance_mask_bool]
        instance_pc_filtered_squeeze = instance_pc_filtered.squeeze(0).view(3, -1).permute(1, 0).reshape(-1, 3)
        density = compute_point_density(point_cloud=instance_pc_filtered_squeeze,r=pc_density_radius)
        
        if density.numel() > 0:
            percentile_value = torch.quantile(density.float(), min_density_ratio)
            preserved_part_mask = density>percentile_value
            instance_pc_filtered_squeeze = instance_pc_filtered_squeeze[preserved_part_mask]
        
        if len(instance_pc_filtered_squeeze)>11:
            instance_pc_filtered_squeeze = Obtain_Cluster_Single_PC(cam_points=instance_pc_filtered_squeeze)
        if instance_pc_filtered_squeeze is not None:
            if instance_pc_filtered_squeeze.shape[0]>1:
                if visible_masks[instance_ind]:
                    ROI_LiDAR_Dict[target_frame_instance_ids[instance_ind]] = instance_pc_filtered_squeeze
                
                else:
                    ROI_LiDAR_Dict[target_frame_instance_ids[instance_ind]] = None
            else:
                ROI_LiDAR_Dict[target_frame_instance_ids[instance_ind]] = None
        else:
            ROI_LiDAR_Dict[target_frame_instance_ids[instance_ind]] = None
    

    raw_point_set_after_clustering = merge_for_visualization_version_1(pcds=ROI_LiDAR_Dict)
    
    
    if len(raw_point_set_after_clustering)>0:
        
        raw_point_set_after_clustering = [tensor.to(image.device) for tensor in raw_point_set_after_clustering]
        raw_point_set_after_clustering = torch.cat(raw_point_set_after_clustering,dim=0)
    

        pc_squeezed = raw_point_set_after_clustering
        projected_valid_mask = point_cloud_to_2d_mask(pc_squeezed.type_as(extrinsic_matrices),
                                                    K=intrinsic_matrices,world_to_camera=extrinsic_matrices,
                                                    H=image.shape[-2],W=image.shape[-1])
        
        projected_valid_mask = projected_valid_mask

        if threeD_visualization:
            visualize_point_cloud_with_axis(point_cloud=pc_squeezed,boxes_3d=boxes_3d_target_frame.cpu().numpy())
    else:
        raw_point_set_after_clustering = None
        projected_valid_mask = None
        
    
    return ROI_LiDAR_Dict, projected_valid_mask


if __name__=="__main__":
    
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    input_example_path = "Debug_Examples/input_example_with_depth.pkl"
    multi_inputs = read_pickle_file(input_example_path)
    # ROI_LiDAR_Dict, projected_valid_mask = Get_Pseudo_Roi_LiDAR(data_inputs=multi_inputs[0],
    #                      target_frame_instance_ids=multi_inputs[0]['instance_ids'][0].cpu().numpy().tolist(),
    #                      threeD_visualization=False)
    


    # for keys in tqdm(multi_inputs.keys()):
    #     ROI_LiDAR_Dict,projected_mask = Get_Pseudo_Roi_LiDAR(data_inputs=multi_inputs[keys],target_frame_instance_ids=multi_inputs[0]['instance_ids'][0].cpu().numpy().tolist(),
    #                         threeD_visualization=False)
        
    #     multi_inputs[keys]["Roi_LiDAR_Dict"] = ROI_LiDAR_Dict
    #     multi_inputs[keys]["projected_valid_mask"] = projected_mask
    
    
    # save_to_pickle("Debug_Examples/example_with_ROI_LIDAR.pkl",multi_inputs)
    
    
    

    
    





