import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import pickle
from tqdm import tqdm
import numpy as np

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def start_with_2013(string):
    return string[string.index("2013_"):]


def Processed_ROI_LiDAR(RoI_box_points):
    
    RoI_points = RoI_box_points['RoI_points']
    
    kepted_RoI_points_List = []
    boxes2d_list = []
    boxes3d_list = []
    
    for idx in range(RoI_box_points['bbox2d'].shape[0]):
        if isinstance(RoI_points[idx],list):
            RoI_points[idx] = np.array(RoI_points[idx])
    
        
        RoI_pd_nums = RoI_points[idx].shape[0]
        if RoI_pd_nums>=3:
            kepted_RoI_points_List.append(RoI_points[idx])
            
            boxes_2d = torch.from_numpy(RoI_box_points['bbox2d'][idx]).unsqueeze(0)
            boxes_3d = (RoI_box_points['bbox3d'][idx]).unsqueeze(0)
            boxes2d_list.append(boxes_2d)
            boxes3d_list.append(boxes_3d)
    
    
    if len(boxes2d_list)==0:
        return None
    else:
        bboxes2d = torch.cat(boxes2d_list,dim=0)
        bboxes3d = torch.cat(boxes3d_list,dim=0)

        RoI_box_points['bbox2d'] = bboxes2d
        RoI_box_points['bbox3d'] = bboxes3d
        RoI_box_points['RoI_points'] = kepted_RoI_points_List

        return RoI_box_points
        



if __name__=="__main__":
    
    train_root_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/training/"
    
    train_filenames = '/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/ImageSets_For_WeakM3D/train.txt'
    train_weight_filepath = '/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/ImageSets_For_WeakM3D/train_weight.txt'
    
    
    train_filenames_list = read_text_lines(train_filenames)
    train_weight_list = read_text_lines(train_weight_filepath)
    
    cnt = 0
    for idx, fname in enumerate(train_filenames_list):
        left_fname = os.path.join(train_root_path,'image_2',fname+".png")
        assert os.path.exists(left_fname)
                
        lidar_RoI_points_path = left_fname.replace("image_2","ROI_LiDAR_InternImage").replace('png', 'pkl')
        assert os.path.exists(lidar_RoI_points_path)

        with open(lidar_RoI_points_path, 'rb') as f:
            RoI_box_points = pickle.load(f) #['bbox2d', 'bbox3d', 'RoI_points']
        
        RoI_box_points = Processed_ROI_LiDAR(RoI_box_points)
        
        if RoI_box_points is not None:
            print(RoI_box_points['bbox2d'].shape)
            print(RoI_box_points['bbox3d'].shape)
            print(len(RoI_box_points['RoI_points']))
                
                
        
        

