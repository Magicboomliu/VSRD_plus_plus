
import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("../..")
from Metrics.eval import get_official_eval_result
from Metrics.kitti_utils import get_objects_from_label
from Metrics.geo_op import rotation_matrix_x


import argparse

import torch
from vsrd.operations.kitti360_operations import box3dIou
from scipy.optimize import linear_sum_assignment
import json


def save_dict_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def find_ones_indices(lst):
    return [index for index, value in enumerate(lst) if value == 1]

def find_zeros_indices(lst):
    return [index for index, value in enumerate(lst) if value == 0]
    
def change_dimension(corners):
    corners[:, [0, 1, 2, 3]], corners[:, [4, 5, 6, 7]] = corners[:, [4, 5, 6, 7]], corners[:, [0, 1, 2, 3]].clone()
    
    return corners

def reorder_list(lst, index):
    return [lst[i] for i in index]

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

def bilateral_matching(boxes1, boxes2,rot):

    iou_matrix = compute_3d_iou_matrix(boxes1, boxes2,rot)

    # # 检查 IoU 矩阵是否包含无效值
    # if torch.any(torch.isnan(iou_matrix)):
    #     print("Error: IoU matrix contains NaN values.")
    #     iou_matrix[torch.isnan(iou_matrix)] = -1e6  # 设置为无效匹配的值

    # if torch.any(torch.isinf(iou_matrix)):
    #     print("Error: IoU matrix contains Inf values.")
    #     iou_matrix[torch.isinf(iou_matrix)] = -1e6  # 设置为无效匹配的值
    
    
    row_indices, col_indices = linear_sum_assignment(-iou_matrix.cpu().numpy())  # 最大化 IoU
    matched_boxes1 = boxes1[row_indices]
    matched_boxes2 = boxes2[col_indices]    
    
    return matched_boxes1, matched_boxes2,row_indices,col_indices

def eval_from_scrach_dynamic(gt_dir, det_dir, ap_mode=40,saved_fname=None):
    global AP_mode
    AP_mode = ap_mode
    all_gt, all_det = [], []
    all_f = sorted(os.listdir(det_dir))
    for i, f in enumerate(tqdm(all_f)):
        gt_f = np.loadtxt(os.path.join(gt_dir, f), dtype=str).reshape(-1, 17)
        dynamic_label_list = gt_f[:,-1].tolist()
        dynamic_label_list = [int(float(value)) for value in dynamic_label_list]
        dynamic_index = find_ones_indices(dynamic_label_list)
        if len(dynamic_index)==0:
            continue
        # # filter out the static labels, only kept the dynamic 
        # gt_f = gt_f[dynamic_index]

        # also do the filtering here at the prediction 3d bounding box    
        det_f = np.loadtxt(os.path.join(det_dir, f), dtype=str).reshape(-1, 16)
        

        # read the pd boxes
        pd_objects = get_objects_from_label(os.path.join(det_dir, f))
        pd_boxes3d_list = []
        for inner_idx, object in enumerate(pd_objects):
            pd_object = pd_objects[inner_idx]
            pd_box3d = pd_object.generate_corners3d()
            pd_boxes3d_list.append(torch.from_numpy(pd_box3d).unsqueeze(0))
        
        if len(pd_boxes3d_list)==0:
            continue
        pd_boxes3d_tensor = torch.cat(pd_boxes3d_list,dim=0) #[N,8,3]
        assert det_f.shape[0]==pd_boxes3d_tensor.shape[0]
        
        # read the gt boxes
        gt_objects = get_objects_from_label(os.path.join(gt_dir, f))
        gt_boxes3d_list = []
        for inner_idx, object in enumerate(gt_objects):
            gt_object = gt_objects[inner_idx]
            gt_box3d = gt_object.generate_corners3d()
            gt_boxes3d_list.append(torch.from_numpy(gt_box3d).unsqueeze(0))
        gt_boxes3d_tensor = torch.cat(gt_boxes3d_list,dim=0) #[N,8,3]
        assert gt_f.shape[0]==gt_boxes3d_tensor.shape[0]
        
        # only filter the dynamic objects
        gt_boxes3d_tensor = gt_boxes3d_tensor[dynamic_index]
        gt_boxes3d_tensor = change_dimension(gt_boxes3d_tensor)
        pd_boxes3d_tensor = change_dimension(pd_boxes3d_tensor)
        gt_boxes3d_tensor = gt_boxes3d_tensor.float()
        pd_boxes3d_tensor = pd_boxes3d_tensor.float()
        rotation_matrix = rotation_matrix_x(torch.tensor(-np.pi / 2.0)).float()
        try:
            gt_boxes3d_tensor, pd_boxes3d_tensor,gt_index,pd_index = bilateral_matching(boxes1=gt_boxes3d_tensor,
                                                                                        boxes2=pd_boxes3d_tensor,
                                                                                        rot=rotation_matrix)
        except:
            continue
        gt_f = gt_f[dynamic_index]
        gt_f = gt_f[gt_index]
        det_f = det_f[pd_index]
        
        
        gt = {}
        det = {}
        '''bbox'''
        gt['bbox'] = gt_f[:, 4:8].astype(np.float32)
        det['bbox'] = det_f[:, 4:8].astype(np.float32)

        '''alpha'''
        gt['alpha'] = gt_f[:, 3].astype(np.float32)
        det['alpha'] = det_f[:, 3].astype(np.float32)

        '''occluded'''
        gt['occluded'] = gt_f[:, 2].astype(np.float32)
        det['occluded'] = det_f[:, 2].astype(np.float32)

        '''truncated'''
        gt['truncated'] = gt_f[:, 1].astype(np.float32)
        det['truncated'] = det_f[:, 1].astype(np.float32)

        '''name'''
        gt['name'] = gt_f[:, 0]
        det['name'] = det_f[:, 0]

        '''location'''
        gt['location'] = gt_f[:, 11:14].astype(np.float32)
        det['location'] = det_f[:, 11:14].astype(np.float32)

        '''dimensions, convert hwl to lhw'''
        gt['dimensions'] = gt_f[:, [10, 8, 9]].astype(np.float32)
        det['dimensions'] = det_f[:, [10, 8, 9]].astype(np.float32)

        '''rotation_y'''
        gt['rotation_y'] = gt_f[:, 14].astype(np.float32)
        det['rotation_y'] = det_f[:, 14].astype(np.float32)

        '''score'''
        det['score'] = det_f[:, 15].astype(np.float32)

        ''' append to tail'''
        all_gt.append(gt)
        all_det.append(det)

    if AP_mode == 40:
        print('-' * 40 + 'AP40 evaluation' + '-' * 40)
    if AP_mode == 11:
        print('-' * 40 + 'AP11 evaluation' + '-' * 40)
    saved_dict = dict()
    
    print('------------------evalute model: {}--------------------'.format(det_dir.split('/')[-2]))
    for cls in ['Car']:
        print('*' * 20 + cls + '*' * 20)
        res = get_official_eval_result(all_gt, all_det, cls, z_axis=1, z_center=1)
        Car_res = res['detail'][cls]
        for k in Car_res.keys():
            # print(k, Car_res[k])
            saved_dict[k] = Car_res[k]
    
    save_dict_to_json(saved_dict,filename=saved_fname)

def eval_from_scrach_static(gt_dir, det_dir, ap_mode=40,saved_fname=None):
    global AP_mode
    AP_mode = ap_mode
    all_gt, all_det = [], []
    all_f = sorted(os.listdir(det_dir))
    for i, f in enumerate(tqdm(all_f)):
        gt_f = np.loadtxt(os.path.join(gt_dir, f), dtype=str).reshape(-1, 17)
        dynamic_label_list = gt_f[:,-1].tolist()
        dynamic_label_list = [int(float(value)) for value in dynamic_label_list]
        dynamic_index = find_zeros_indices(dynamic_label_list)
        if len(dynamic_index)==0:
            continue

        # also do the filtering here at the prediction 3d bounding box    
        det_f = np.loadtxt(os.path.join(det_dir, f), dtype=str).reshape(-1, 16)
        
        # read the pd boxes
        pd_objects = get_objects_from_label(os.path.join(det_dir, f))
        pd_boxes3d_list = []
        for inner_idx, object in enumerate(pd_objects):
            pd_object = pd_objects[inner_idx]
            pd_box3d = pd_object.generate_corners3d()
            pd_boxes3d_list.append(torch.from_numpy(pd_box3d).unsqueeze(0))
        
        if len(pd_boxes3d_list)==0:
            continue
        
        pd_boxes3d_tensor = torch.cat(pd_boxes3d_list,dim=0) #[N,8,3]
        assert det_f.shape[0]==pd_boxes3d_tensor.shape[0]
        
        # read the gt boxes
        gt_objects = get_objects_from_label(os.path.join(gt_dir, f))
        gt_boxes3d_list = []
        for inner_idx, object in enumerate(gt_objects):
            gt_object = gt_objects[inner_idx]
            gt_box3d = gt_object.generate_corners3d()
            gt_boxes3d_list.append(torch.from_numpy(gt_box3d).unsqueeze(0))
        gt_boxes3d_tensor = torch.cat(gt_boxes3d_list,dim=0) #[N,8,3]
        assert gt_f.shape[0]==gt_boxes3d_tensor.shape[0]
        
        # only filter the dynamic objects
        gt_boxes3d_tensor = gt_boxes3d_tensor[dynamic_index]
        gt_boxes3d_tensor = change_dimension(gt_boxes3d_tensor)
        pd_boxes3d_tensor = change_dimension(pd_boxes3d_tensor)
        gt_boxes3d_tensor = gt_boxes3d_tensor.float()
        pd_boxes3d_tensor = pd_boxes3d_tensor.float()
        rotation_matrix = rotation_matrix_x(torch.tensor(-np.pi / 2.0)).float()
        try:
            gt_boxes3d_tensor, pd_boxes3d_tensor,gt_index,pd_index = bilateral_matching(boxes1=gt_boxes3d_tensor,
                                                                                    boxes2=pd_boxes3d_tensor,
                                                                                rot=rotation_matrix)
        except:
            continue
        gt_f = gt_f[dynamic_index]
        gt_f = gt_f[gt_index]
        det_f = det_f[pd_index]
        
        
        gt = {}
        det = {}
        '''bbox'''
        gt['bbox'] = gt_f[:, 4:8].astype(np.float32)
        det['bbox'] = det_f[:, 4:8].astype(np.float32)

        '''alpha'''
        gt['alpha'] = gt_f[:, 3].astype(np.float32)
        det['alpha'] = det_f[:, 3].astype(np.float32)

        '''occluded'''
        gt['occluded'] = gt_f[:, 2].astype(np.float32)
        det['occluded'] = det_f[:, 2].astype(np.float32)

        '''truncated'''
        gt['truncated'] = gt_f[:, 1].astype(np.float32)
        det['truncated'] = det_f[:, 1].astype(np.float32)

        '''name'''
        gt['name'] = gt_f[:, 0]
        det['name'] = det_f[:, 0]

        '''location'''
        gt['location'] = gt_f[:, 11:14].astype(np.float32)
        det['location'] = det_f[:, 11:14].astype(np.float32)

        '''dimensions, convert hwl to lhw'''
        gt['dimensions'] = gt_f[:, [10, 8, 9]].astype(np.float32)
        det['dimensions'] = det_f[:, [10, 8, 9]].astype(np.float32)

        '''rotation_y'''
        gt['rotation_y'] = gt_f[:, 14].astype(np.float32)
        det['rotation_y'] = det_f[:, 14].astype(np.float32)

        '''score'''
        det['score'] = det_f[:, 15].astype(np.float32)

        ''' append to tail'''
        all_gt.append(gt)
        all_det.append(det)

    if AP_mode == 40:
        print('-' * 40 + 'AP40 evaluation' + '-' * 40)
    if AP_mode == 11:
        print('-' * 40 + 'AP11 evaluation' + '-' * 40)

    saved_dict = dict()
    
    print('------------------evalute model: {}--------------------'.format(det_dir.split('/')[-2]))
    for cls in ['Car']:
        print('*' * 20 + cls + '*' * 20)
        res = get_official_eval_result(all_gt, all_det, cls, z_axis=1, z_center=1)
        Car_res = res['detail'][cls]
        for k in Car_res.keys():
            # print(k, Car_res[k])
            saved_dict[k] = Car_res[k]
    
    save_dict_to_json(saved_dict,filename=saved_fname)
    
    
def eval_from_scrach_static_V2(gt_dir, det_dir, ap_mode=40,saved_fname=None):
    global AP_mode
    AP_mode = ap_mode
    all_gt, all_det = [], []
    all_f = sorted(os.listdir(det_dir))
    cnt = 0
    for i, f in enumerate(tqdm(all_f)):
        gt_f = np.loadtxt(os.path.join(gt_dir, f), dtype=str).reshape(-1, 17)
        dynamic_label_list = gt_f[:,-1].tolist()
        dynamic_label_list = [int(float(value)) for value in dynamic_label_list]
        dynamic_index = find_zeros_indices(dynamic_label_list)
        dynamic_index2 = find_ones_indices(dynamic_label_list)
        
        if len(dynamic_index2)>0:
            cnt = cnt +1
            continue
        
        if len(dynamic_index)==0:
            continue

        # also do the filtering here at the prediction 3d bounding box    
        det_f = np.loadtxt(os.path.join(det_dir, f), dtype=str).reshape(-1, 16)
        
        # # read the pd boxes
        # pd_objects = get_objects_from_label(os.path.join(det_dir, f))
        # pd_boxes3d_list = []
        # for inner_idx, object in enumerate(pd_objects):
        #     pd_object = pd_objects[inner_idx]
        #     pd_box3d = pd_object.generate_corners3d()
        #     pd_boxes3d_list.append(torch.from_numpy(pd_box3d).unsqueeze(0))
        
        # if len(pd_boxes3d_list)==0:
        #     continue
        
        # pd_boxes3d_tensor = torch.cat(pd_boxes3d_list,dim=0) #[N,8,3]
        # assert det_f.shape[0]==pd_boxes3d_tensor.shape[0]
        
        # # read the gt boxes
        # gt_objects = get_objects_from_label(os.path.join(gt_dir, f))
        # gt_boxes3d_list = []
        # for inner_idx, object in enumerate(gt_objects):
        #     gt_object = gt_objects[inner_idx]
        #     gt_box3d = gt_object.generate_corners3d()
        #     gt_boxes3d_list.append(torch.from_numpy(gt_box3d).unsqueeze(0))
        # gt_boxes3d_tensor = torch.cat(gt_boxes3d_list,dim=0) #[N,8,3]
        # assert gt_f.shape[0]==gt_boxes3d_tensor.shape[0]
        
        # # only filter the dynamic objects
        # gt_boxes3d_tensor = gt_boxes3d_tensor[dynamic_index]
        # gt_boxes3d_tensor = change_dimension(gt_boxes3d_tensor)
        # pd_boxes3d_tensor = change_dimension(pd_boxes3d_tensor)
        # gt_boxes3d_tensor = gt_boxes3d_tensor.float()
        # pd_boxes3d_tensor = pd_boxes3d_tensor.float()
        # rotation_matrix = rotation_matrix_x(torch.tensor(-np.pi / 2.0)).float()
        # try:
        #     gt_boxes3d_tensor, pd_boxes3d_tensor,gt_index,pd_index = bilateral_matching(boxes1=gt_boxes3d_tensor,
        #                                                                             boxes2=pd_boxes3d_tensor,
        #                                                                         rot=rotation_matrix)
        # except:
        #     continue
        # gt_f = gt_f[dynamic_index]
        # gt_f = gt_f[gt_index]
        # det_f = det_f[pd_index]
        
        
        gt = {}
        det = {}
        '''bbox'''
        gt['bbox'] = gt_f[:, 4:8].astype(np.float32)
        det['bbox'] = det_f[:, 4:8].astype(np.float32)

        '''alpha'''
        gt['alpha'] = gt_f[:, 3].astype(np.float32)
        det['alpha'] = det_f[:, 3].astype(np.float32)

        '''occluded'''
        gt['occluded'] = gt_f[:, 2].astype(np.float32)
        det['occluded'] = det_f[:, 2].astype(np.float32)

        '''truncated'''
        gt['truncated'] = gt_f[:, 1].astype(np.float32)
        det['truncated'] = det_f[:, 1].astype(np.float32)

        '''name'''
        gt['name'] = gt_f[:, 0]
        det['name'] = det_f[:, 0]

        '''location'''
        gt['location'] = gt_f[:, 11:14].astype(np.float32)
        det['location'] = det_f[:, 11:14].astype(np.float32)

        '''dimensions, convert hwl to lhw'''
        gt['dimensions'] = gt_f[:, [10, 8, 9]].astype(np.float32)
        det['dimensions'] = det_f[:, [10, 8, 9]].astype(np.float32)

        '''rotation_y'''
        gt['rotation_y'] = gt_f[:, 14].astype(np.float32)
        det['rotation_y'] = det_f[:, 14].astype(np.float32)

        '''score'''
        det['score'] = det_f[:, 15].astype(np.float32)

        ''' append to tail'''
        all_gt.append(gt)
        all_det.append(det)

    if AP_mode == 40:
        print('-' * 40 + 'AP40 evaluation' + '-' * 40)
    if AP_mode == 11:
        print('-' * 40 + 'AP11 evaluation' + '-' * 40)

    saved_dict = dict()
    
    print(cnt,len(all_det),len(all_f))
    
    print('------------------evalute model: {}--------------------'.format(det_dir.split('/')[-2]))
    for cls in ['Car']:
        print('*' * 20 + cls + '*' * 20)
        res = get_official_eval_result(all_gt, all_det, cls, z_axis=1, z_center=1)
        Car_res = res['detail'][cls]
        for k in Car_res.keys():
            # print(k, Car_res[k])
            saved_dict[k] = Car_res[k]
    
    save_dict_to_json(saved_dict,filename=saved_fname)
    
def eval_from_scrach(gt_dir, det_dir, ap_mode=40,saved_fname=None):
    global AP_mode
    AP_mode = ap_mode
    all_gt, all_det = [], []
    all_f = sorted(os.listdir(det_dir))
    for i, f in enumerate(tqdm(all_f)):
        try:
            gt_f = np.loadtxt(os.path.join(gt_dir, f), dtype=str).reshape(-1, 17)
        except:
            gt_f = np.loadtxt(os.path.join(gt_dir, f), dtype=str).reshape(-1, 16)
            
        det_f = np.loadtxt(os.path.join(det_dir, f), dtype=str).reshape(-1, 16)
        
        gt = {}
        det = {}
        '''bbox'''
        gt['bbox'] = gt_f[:, 4:8].astype(np.float32)
        det['bbox'] = det_f[:, 4:8].astype(np.float32)

        '''alpha'''
        gt['alpha'] = gt_f[:, 3].astype(np.float32)
        det['alpha'] = det_f[:, 3].astype(np.float32)

        '''occluded'''
        gt['occluded'] = gt_f[:, 2].astype(np.float32)
        det['occluded'] = det_f[:, 2].astype(np.float32)

        '''truncated'''
        gt['truncated'] = gt_f[:, 1].astype(np.float32)
        det['truncated'] = det_f[:, 1].astype(np.float32)

        '''name'''
        gt['name'] = gt_f[:, 0]
        det['name'] = det_f[:, 0]

        '''location'''
        gt['location'] = gt_f[:, 11:14].astype(np.float32)
        det['location'] = det_f[:, 11:14].astype(np.float32)

        '''dimensions, convert hwl to lhw'''
        gt['dimensions'] = gt_f[:, [10, 8, 9]].astype(np.float32)
        det['dimensions'] = det_f[:, [10, 8, 9]].astype(np.float32)

        '''rotation_y'''
        gt['rotation_y'] = gt_f[:, 14].astype(np.float32)
        det['rotation_y'] = det_f[:, 14].astype(np.float32)

        '''score'''
        det['score'] = det_f[:, 15].astype(np.float32)

        ''' append to tail'''
        all_gt.append(gt)
        all_det.append(det)

    if AP_mode == 40:
        print('-' * 40 + 'AP40 evaluation' + '-' * 40)
    if AP_mode == 11:
        print('-' * 40 + 'AP11 evaluation' + '-' * 40)

    saved_dict = dict()
    
    print('------------------evalute model: {}--------------------'.format(det_dir.split('/')[-2]))
    for cls in ['Car']:
        print('*' * 20 + cls + '*' * 20)
        res = get_official_eval_result(all_gt, all_det, cls, z_axis=1, z_center=1)
        Car_res = res['detail'][cls]
        for k in Car_res.keys():
            # print(k, Car_res[k])
            saved_dict[k] = Car_res[k]
    
    save_dict_to_json(saved_dict,filename=saved_fname)
    
    

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="mAP Computation")
    parser.add_argument("--pd_dir_folder",type=str,default="/media/zliu/data12/dataset/KITTI3D_VSRD_Vanilla/training/label_est/")
    parser.add_argument("--gt_dir_folder",type=str,default="/media/zliu/data12/dataset/KITTI3D_VSRD_Vanilla/training/label_2/")
    parser.add_argument("--saved_mAP_folder",type=str,default="/media/zliu/data12/dataset/KITTI3D_VSRD_Vanilla/training/mAP_Results")
    args = parser.parse_args()
    
    
    pd_dir_folder = args.pd_dir_folder
    gt_dir_folder = args.gt_dir_folder
    os.makedirs(args.saved_mAP_folder,exist_ok=True)
    
    saved_dynamic_mAP_json = os.path.join(args.saved_mAP_folder,'dynamic_mAP.json')
    saved_static_mAP_json = os.path.join(args.saved_mAP_folder,'static_mAP.json')
    saved_whole_mAP_json = os.path.join(args.saved_mAP_folder,'whole_mAP.json')
    
    
    # eval_from_scrach_dynamic(gt_dir=gt_dir_folder,
    #                          det_dir=pd_dir_folder,ap_mode=40,
    #                          saved_fname=saved_dynamic_mAP_json)

    eval_from_scrach_static(gt_dir=gt_dir_folder,
                             det_dir=pd_dir_folder,ap_mode=40,
                             saved_fname=saved_static_mAP_json)
        
    # eval_from_scrach(gt_dir=gt_dir_folder,
    #                          det_dir=pd_dir_folder,ap_mode=40,
    #                          saved_fname=saved_whole_mAP_json)
    