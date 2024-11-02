import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
import sys

import os
import numpy as np
from kitti_utils import get_calib_from_file, get_objects_from_label
from PIL import Image
import matplotlib.pyplot as plt
from draw_projected_3d import draw3d_bbox_2d_projection
from kitti_box_computation import project_to_image

import pycocotools.mask
import torch
import json
import skimage.io
from geo_op import rotation_matrix_x,rotation_matrix_y,rotation_matrix_z
import numpy as np
from scipy.optimize import linear_sum_assignment

from vsrd.operations.kitti360_operations import box3dIou


def change_dimension(corners):
    corners[:, [0, 1, 2, 3]], corners[:, [4, 5, 6, 7]] = corners[:, [4, 5, 6, 7]], corners[:, [0, 1, 2, 3]].clone()
    
    return corners

def reorder_list(lst, index):
    return [lst[i] for i in index]
    
def save_dict_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def main(args):
    
    prediction_folder = args.prediction_folder
    gt_folder = args.gt_folder
    output_name = args.output_name
    
    os.makedirs(output_name,exist_ok=True)
    
    sequences = os.listdir(prediction_folder)
    
    global_3d_iou_all = 0
    global_bev_iou_all = 0
    global_3d_iou_static = 0
    global_bev_iou_static = 0
    global_3d_iou_dynamic = 0
    global_bev_iou_dynamic = 0
    
    
    global_counter = 0
    global_counter_static = 0
    global_counter_dynamic = 0
    
    
    a = 0
    
    for seq_name in sequences:
        
        mean_3d_iou_seq_all = 0
        mean_bev_iou_seq_all = 0
        mean_3d_iou_seq_static = 0
        mean_bev_iou_seq_static = 0
        mean_3d_iou_seq_dynamic = 0
        mean_bev_iou_seq_dynamic = 0
        
        
        
        seq_counter_all = 0
        seq_counter_static = 0
        seq_counter_dynamic = 0
        
        pd_seq_folder = os.path.join(prediction_folder,seq_name,"image_00/data_rect/")
        gt_seq_folder = os.path.join(gt_folder,seq_name,"image_00/data_rect/")
        
        for idx, fname in enumerate(os.listdir(pd_seq_folder)):
            pd_label_path = os.path.join(pd_seq_folder,fname)
            gt_label_path_with_dynamic = os.path.join(gt_seq_folder,fname)
            
            assert os.path.exists(pd_label_path)
            assert os.path.exists(gt_label_path_with_dynamic)
            
    
            pd_objects = get_objects_from_label(pd_label_path)
            pd_boxes3d_list = []
            for inner_idx, object in enumerate(pd_objects):
                pd_object = pd_objects[inner_idx]
                pd_box3d = pd_object.generate_corners3d()
                pd_boxes3d_list.append(torch.from_numpy(pd_box3d).unsqueeze(0))
            pd_boxes3d_tensor = torch.cat(pd_boxes3d_list,dim=0)
            
            
            gt_objects = get_objects_from_label(gt_label_path_with_dynamic)
            gt_dynamic_list = []
            gt_boxes3d_list = []
            difficulity_list = [] # 0 for easy, 1 for hard, -1 for else
            
            for inner_idx, object in enumerate(gt_objects):
                gt_object = gt_objects[inner_idx]
                gt_dynamic_label = gt_object.dynamic_label
                gt_dynamic_list.append(gt_dynamic_label) # zero for "static", one for "dynamic"
                gt_box3d = gt_object.generate_corners3d()
                gt_boxes3d_list.append(torch.from_numpy(gt_box3d).unsqueeze(0))
                gt_box2d = gt_object.box2d
                height = gt_box2d[-1] - gt_box2d[-3]
                if height>40:
                    difficulity_list.append(0)
                elif height<=40 and height>25:
                    difficulity_list.append(1)
                elif height<=25:
                    difficulity_list.append(-1)
            
                
            gt_boxes3d_tensor = torch.cat(gt_boxes3d_list,dim=0) # [N,8,3]
            
            gt_boxes3d_tensor = change_dimension(gt_boxes3d_tensor)
            pd_boxes3d_tensor = change_dimension(pd_boxes3d_tensor)
            
            gt_boxes3d_tensor = gt_boxes3d_tensor.float()
            pd_boxes3d_tensor = pd_boxes3d_tensor.float()
            
            rotation_matrix = rotation_matrix_x(torch.tensor(-np.pi / 2.0)).float()
            
            gt_boxes3d_tensor, pd_boxes3d_tensor,gt_dynamic_list,difficulity_list = bilateral_matching(boxes1=gt_boxes3d_tensor,boxes2=pd_boxes3d_tensor,rot=rotation_matrix,dynamic_list=gt_dynamic_list,
                                                                                      difficulity_list=difficulity_list)
            assert gt_boxes3d_tensor.shape == pd_boxes3d_tensor.shape
            
            

            
            for insatnce_id_inside_image in range(gt_boxes3d_tensor.shape[0]):
                gt_box3d = gt_boxes3d_tensor[insatnce_id_inside_image]
                pd_box3d = pd_boxes3d_tensor[insatnce_id_inside_image]
                current_dyanmic_label = int(float(gt_dynamic_list[insatnce_id_inside_image]))
                current_difficult_label = int(float(difficulity_list[insatnce_id_inside_image]))
       
                
                if current_difficult_label == -1:
                    continue

                if args.options=='easy':
                    if current_difficult_label!=0:
                        continue
                    
                if args.options=='hard':
                    if current_difficult_label!=1:
                        continue
            
                iou3d, iou_bev = box3dIou((gt_box3d@rotation_matrix.T).cpu().numpy(),
                                          (pd_box3d@rotation_matrix.T).cpu().numpy())
                

                # already update all
                global_counter = global_counter +1
                seq_counter_all = seq_counter_all +1
                
                
                mean_3d_iou_seq_all+=iou3d
                mean_bev_iou_seq_all+=iou_bev
                
                global_3d_iou_all+=iou3d
                global_bev_iou_all+=iou_bev
                
                
                # static                
                if current_dyanmic_label==0:
                    global_counter_static = global_counter_static + 1
                    seq_counter_static = seq_counter_static + 1
                    mean_3d_iou_seq_static+=iou3d
                    mean_bev_iou_seq_static+=iou_bev
                    global_3d_iou_static+=iou3d
                    global_bev_iou_static+=iou_bev
                    
                
                # dynamic 
                else:
                    global_counter_dynamic = global_counter_dynamic + 1
                    seq_counter_dynamic = seq_counter_dynamic +1

                    mean_3d_iou_seq_dynamic+=iou3d
                    mean_bev_iou_seq_dynamic+=iou_bev
                    global_3d_iou_dynamic+=iou3d
                    global_bev_iou_dynamic+=iou_bev
                
            

        # static evaluation
        if seq_counter_static==0:
            mean_3d_iou_seq_static = None
            mean_bev_iou_seq_static = None
        else:
            mean_3d_iou_seq_static = mean_3d_iou_seq_static/seq_counter_static
            mean_bev_iou_seq_static = mean_bev_iou_seq_static/seq_counter_static

        # dynamic evalutaion
        if seq_counter_dynamic==0:
            mean_3d_iou_seq_dynamic = None
            mean_bev_iou_seq_dynamic = None
        else:
            mean_3d_iou_seq_dynamic = mean_3d_iou_seq_dynamic/seq_counter_dynamic
            mean_bev_iou_seq_dynamic = mean_bev_iou_seq_dynamic/seq_counter_dynamic
        
        # all evaluation
        mean_3d_iou_seq_all = mean_3d_iou_seq_all/seq_counter_all
        mean_bev_iou_seq_all = mean_bev_iou_seq_all/seq_counter_all
        
        assert seq_counter_all == (seq_counter_dynamic+seq_counter_static)
        
        seq_IoU_result_folder = os.path.join(output_name,seq_name)
        os.makedirs(seq_IoU_result_folder,exist_ok=True)
        seq_iou_result_name = os.path.join(seq_IoU_result_folder,"iou_{}.json".format(args.options))
        
        seq_result_dict = dict()
        
        
        seq_result_dict['IoU@3D-All'] = mean_3d_iou_seq_all
        seq_result_dict['IoU@3D-Static'] = mean_3d_iou_seq_static
        seq_result_dict['IoU@3D-Dynamic'] = mean_3d_iou_seq_dynamic

        seq_result_dict['IoU@BEV-All'] = mean_bev_iou_seq_all
        seq_result_dict['IoU@BEV-Static'] = mean_bev_iou_seq_static
        seq_result_dict['IoU@BEV-Dynamic'] = mean_bev_iou_seq_dynamic
        
        
        save_dict_to_json(data=seq_result_dict,filename=seq_iou_result_name)
        
    
    # staic        
    if global_counter_static==0:
        global_3d_iou_static = None
        global_bev_iou_static = None
    else:
        global_3d_iou_static = global_3d_iou_static/global_counter_static
        global_bev_iou_static = global_bev_iou_static/global_counter_static
    
    # dynamic 
    if global_counter_dynamic==0:
        global_3d_iou_dynamic = None
        global_bev_iou_dynamic = None
    else:
        global_3d_iou_dynamic = global_3d_iou_dynamic/global_counter_dynamic
        global_bev_iou_dynamic = global_bev_iou_dynamic/global_counter_dynamic
    
    # all
    global_3d_iou_all = global_3d_iou_all/global_counter
    global_bev_iou_all = global_bev_iou_all/global_counter
    
    assert global_counter == (global_counter_dynamic + global_counter_static)


    seq_IoU_result_folder = output_name
    seq_iou_result_name = os.path.join(seq_IoU_result_folder,"iou_{}.json".format(args.options))
    
    seq_result_dict = dict()
    
    
    seq_result_dict['IoU@3D-All'] = global_3d_iou_all
    seq_result_dict['IoU@3D-Static'] = global_3d_iou_static
    seq_result_dict['IoU@3D-Dynamic'] = global_3d_iou_dynamic

    seq_result_dict['IoU@BEV-All'] = global_bev_iou_all
    seq_result_dict['IoU@BEV-Static'] = global_bev_iou_static
    seq_result_dict['IoU@BEV-Dynamic'] = global_bev_iou_dynamic

    save_dict_to_json(data=seq_result_dict,filename=seq_iou_result_name)
    




        
        
    
    
        

            

def bilateral_matching(boxes1, boxes2,dynamic_list,rot,difficulity_list):

    iou_matrix = compute_3d_iou_matrix(boxes1, boxes2,rot)
    row_indices, col_indices = linear_sum_assignment(-iou_matrix.cpu().numpy())  # 最大化 IoU
    matched_boxes1 = boxes1[row_indices]
    matched_boxes2 = boxes2[col_indices]
    dynamic_list = reorder_list(lst=dynamic_list,index=row_indices)
    
    difficulity_list = reorder_list(lst=difficulity_list,index=row_indices)
    
    return matched_boxes1, matched_boxes2,dynamic_list,difficulity_list


def compute_3d_iou_matrix(boxes1, boxes2,rot):

    N1, N2 = boxes1.shape[0], boxes2.shape[0]
    iou_matrix = torch.zeros((N1, N2), device=boxes1.device)

    for i in range(N1):
        for j in range(N2):
            iou_matrix[i, j] = compute_3d_iou(boxes1[i]@rot.T, boxes2[j]@rot.T)
    
    return iou_matrix


def compute_3d_iou(box1, box2):

    iou, _ = box3dIou(box1.cpu().numpy(), box2.cpu().numpy())
    return torch.tensor(iou, device=box1.device)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Stage2 Visualizations Projected 3D Boxes")
    parser.add_argument("--prediction_folder",type=str,default="datasets/KITTI-360")
    parser.add_argument("--gt_folder",type=str,default="datasets/KITTI-360")
    parser.add_argument("--output_name",type=str, default="datasets/KITTI-360")
    parser.add_argument("--options",type=str, default="datasets/KITTI-360") # easy, hard, all
    args = parser.parse_args()

    
    main(args=args)
