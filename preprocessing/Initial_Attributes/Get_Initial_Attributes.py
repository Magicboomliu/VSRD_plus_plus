import sys
sys.path.append("../..")
import os
from preprocessing.Initial_Attributes.file_io_utils import read_pickle_file,read_depth,get_depth_filename,save_to_pickle,visualize_point_cloud_with_axis_with_two_box
from preprocessing.Initial_Attributes.Get_GT_Attribute import get_gt_location_velocity_orientation
from preprocessing.Initial_Attributes.Get_Instance_LiDAR import Get_Pseudo_Roi_LiDAR
from preprocessing.Initial_Attributes.Get_Velocity import Get_Estimated_Velocity_Using_CPI
from preprocessing.Initial_Attributes.Get_Location_Orientation import get_location_orientation

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import time


def Update_Multi_Inputs(multi_inputs):
    for key in multi_inputs.keys():
        
        left_image_path = multi_inputs[key]['filenames'][0]
        left_image_data = multi_inputs[key]['images']
        # MAKE SURE the Depth is exited.        


        assert os.path.exists(get_depth_filename(left_image_path))
        left_depth_data = torch.from_numpy(read_depth(get_depth_filename(left_image_path))).unsqueeze(0).unsqueeze(0).to(left_image_data.device)
        multi_inputs[key]['pseudo_depth'] = left_depth_data
    
    return multi_inputs



def Get_Initial_Attributes(multi_inputs,dynamic_mask_list,device="cuda:0"):
    multi_inputs = Update_Multi_Inputs(multi_inputs)

    # Step 1: Get the RoI Lidar Points
    for keys in tqdm(multi_inputs.keys()):
        ROI_LiDAR_Dict,projected_mask = Get_Pseudo_Roi_LiDAR(data_inputs=multi_inputs[keys],
                                                             target_frame_instance_ids=multi_inputs[0]['instance_ids'][0].cpu().numpy().tolist(),
                            threeD_visualization=False)
        
        # there is a possibility that the project_mask is None
        if projected_mask == None:
            multi_inputs[keys]['LiDAR_Validality'] = False  # This ROI LiDAR is use-less
        else:
            multi_inputs[keys]['LiDAR_Validality'] = True # This ROI LiDAR is can be used
        
        multi_inputs[keys]["Roi_LiDAR_Dict"] = ROI_LiDAR_Dict
        multi_inputs[keys]["projected_valid_mask"] = projected_mask
        
    
    
    # Step2: Get the Velocity
    '''Get the Estimated Velocity'''
    estimated_velocity = Get_Estimated_Velocity_Using_CPI(multi_inputs=multi_inputs,dynamic_mask=dynamic_mask_list,device=device)
    for key in multi_inputs.keys():
        multi_inputs[key]["velo"] = estimated_velocity


    # Step3: Get the Estimation Location and the Reference Orientation(at target frame)
    multi_inputs = get_location_orientation(multi_inputs=multi_inputs,dynamic_mask=dynamic_mask_list,threshold=120,
                                            device=device)

    try:
        # Step4: Get the GT Ones just for debugging
        locations,dimensions,orientations,gt_velocity = get_gt_location_velocity_orientation(multi_inputs=multi_inputs,
                                                                                dyanmic_mask_list=dynamic_mask_list,
                                                                                use_velocity_direction=True)

        multi_inputs[0]['gt_loc'] = locations
        multi_inputs[0]['gt_dimension'] = dimensions
        multi_inputs[0]['gt_orientation'] = orientations
        multi_inputs[0]['gt_velo'] = gt_velocity
    except:
        multi_inputs[0]['gt_loc'] = None
        multi_inputs[0]['gt_dimension'] = None
        multi_inputs[0]['gt_orientation'] = None
        multi_inputs[0]['gt_velo'] = None
    
    
    
    
    return multi_inputs
    
    


if __name__=="__main__":

    data_file = "/home/zliu/CVPR2025/Test_Examples/exampleV2.pkl"
    multi_inputs = read_pickle_file(data_file)
    
    dynamic_mask_list = [False,False,False,True,True,True]
    multi_inputs = Get_Initial_Attributes(multi_inputs=multi_inputs,dynamic_mask_list=dynamic_mask_list,
                                          device="cuda:0")
    
    
    
    print(multi_inputs[0]['est_loc'])
    print(multi_inputs[0]['velo'])
    print("-----------------------------------")
    print(multi_inputs[0]['gt_loc'])
    print(multi_inputs[0]['gt_velo'])
    
    # gt_orientation = multi_inputs[0]['gt_orientation']
    # gt_dimension = multi_inputs[0]['gt_dimension']
    # estimated_orientation = multi_inputs[0]['est_orient']
    
    # est_lidars = multi_inputs[0]['est_lidars']
    
    
    # est_lidars_all = torch.cat(est_lidars,dim=0)
    
    
    # from Optimized_Based.utils.box_geo import decode_box_3d
    # from preprocessing.Initial_Attributes.file_io_utils import visualize_point_cloud_with_axis_with_two_box_with_pcd


    # align_bounding_box = decode_box_3d(locations=multi_inputs[0]['est_loc'].to(gt_dimension.device),
    #                                     dimensions=gt_dimension,
    #                                     orientations=multi_inputs[0]['est_orient'].to(gt_dimension.device))
    
    
    # gt_bounding_box = decode_box_3d(locations=multi_inputs[0]['gt_loc'].to(gt_dimension.device),
    #                                     dimensions=gt_dimension,
    #                                     orientations=gt_orientation.to(gt_dimension.device))
    
    
    # visualize_point_cloud_with_axis_with_two_box_with_pcd(
    #     axis_vis=True,boxes_3d1=align_bounding_box[0].cpu().numpy(),boxes_3d2=gt_bounding_box[0].cpu().numpy(),
    #     point_cloud=est_lidars_all)







    



