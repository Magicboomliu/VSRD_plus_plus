
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



from scipy.optimize import linear_sum_assignment

def compute_iou_matrix(boxes_A, boxes_B):
    """
    计算两个边界框集合之间的 IoU 矩阵。
    参数:
        boxes_A: shape (N1, 4)，边界框集合 A
        boxes_B: shape (N2, 4)，边界框集合 B
    返回:
        iou_matrix: shape (N1, N2)，A 和 B 之间的 IoU 矩阵。
    """
    N1, N2 = len(boxes_A), len(boxes_B)
    iou_matrix = np.zeros((N1, N2))

    for i, box_a in enumerate(boxes_A):
        for j, box_b in enumerate(boxes_B):
            x1 = max(box_a[0], box_b[0])
            y1 = max(box_a[1], box_b[1])
            x2 = min(box_a[2], box_b[2])
            y2 = min(box_a[3], box_b[3])

            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            union_area = area_a + area_b - inter_area + 1e-6

            iou_matrix[i, j] = inter_area / union_area

    return iou_matrix

def bilateral_matching_linear_sum_assignment(boxes_A, boxes_B):
    """
    使用 linear_sum_assignment 进行 IoU 最大化的双向匹配。
    参数:
        boxes_A: shape (N1, 4)，边界框集合 A
        boxes_B: shape (N2, 4)，边界框集合 B
    返回:
        matched_index_A: list，匹配的 A 中的索引
        matched_index_B: list，匹配的 B 中的索引
    """
    # 计算 IoU 矩阵
    iou_matrix = compute_iou_matrix(boxes_A, boxes_B)

    # 将 IoU 转化为损失矩阵（负值或反向变换）
    cost_matrix = 1 - iou_matrix

    # 使用 linear_sum_assignment 求解最小成本匹配
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # 筛选有效的匹配对（IoU > 0 的部分）
    matched_index_A = []
    matched_index_B = []
    for i, j in zip(row_indices, col_indices):
        if iou_matrix[i, j] > 0:  # 忽略 IoU 为 0 的匹配
            matched_index_A.append(i)
            matched_index_B.append(j)

    return matched_index_A, matched_index_B





def save_dict_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def find_ones_indices(lst):
    return [index for index, value in enumerate(lst) if value == 1]

def find_zeros_indices(lst):
    return [index for index, value in enumerate(lst) if value == 0]
    


def eval_from_scrach_dynamic(gt_dir, det_dir, ap_mode=40,saved_fname=None):
    global AP_mode
    AP_mode = ap_mode
    all_gt, all_det = [], []
    all_f = sorted(os.listdir(det_dir))
    for i, f in enumerate(tqdm(all_f)):
        gt_f = np.loadtxt(os.path.join(gt_dir, f), dtype=str).reshape(-1, 17)
        
        # also do the filtering here at the prediction 3d bounding box    
        det_f = np.loadtxt(os.path.join(det_dir, f), dtype=str).reshape(-1, 16)
        
        det_f_box2d = det_f[:,4:8].astype(np.float32)
        gt_f_box2d = gt_f[:,4:8].astype(np.float32)
        
        index_det, index_gt = bilateral_matching_linear_sum_assignment(boxes_A=det_f_box2d,boxes_B=gt_f_box2d)
        
        det_f = det_f[index_det]
        gt_f = gt_f[index_gt]

        # get dynamic box
        dynamic_label_list = gt_f[:,-1].tolist()
        dynamic_label_list = [int(float(value)) for value in dynamic_label_list]
        dynamic_index = find_ones_indices(dynamic_label_list)  # aall the dynamic index from the gt labels
        if len(dynamic_index)==0:
            continue
        
        # only the dynamic objects
        gt_f = gt_f[dynamic_index]
        det_f = det_f[dynamic_index]
        
        
        
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
        
        # also do the filtering here at the prediction 3d bounding box    
        det_f = np.loadtxt(os.path.join(det_dir, f), dtype=str).reshape(-1, 16)
        
        det_f_box2d = det_f[:,4:8].astype(np.float32)
        gt_f_box2d = gt_f[:,4:8].astype(np.float32)
        
        index_det, index_gt = bilateral_matching_linear_sum_assignment(boxes_A=det_f_box2d,boxes_B=gt_f_box2d)
        
        det_f = det_f[index_det]
        gt_f = gt_f[index_gt]
        
        
        
        dynamic_label_list = gt_f[:,-1].tolist()
        dynamic_label_list = [int(float(value)) for value in dynamic_label_list]
        dynamic_index = find_zeros_indices(dynamic_label_list)
        if len(dynamic_index)==0:
            continue

        # only the static objects
        gt_f = gt_f[dynamic_index]
        det_f = det_f[dynamic_index]

    
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
    
    
    eval_from_scrach_dynamic(gt_dir=gt_dir_folder,
                             det_dir=pd_dir_folder,ap_mode=40,
                             saved_fname=saved_dynamic_mAP_json)

    eval_from_scrach_static(gt_dir=gt_dir_folder,
                             det_dir=pd_dir_folder,ap_mode=40,
                             saved_fname=saved_static_mAP_json)

    
    eval_from_scrach(gt_dir=gt_dir_folder,
                             det_dir=pd_dir_folder,ap_mode=40,
                             saved_fname=saved_whole_mAP_json)
        
