import os
import random
from tqdm import tqdm
import shutil
import argparse
import numpy as np

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


def found_the_index(new_data,old_data):
    new_data_box2d = new_data[:,4:8].astype(np.float32)
    old_data_box2d = old_data[:,4:8].astype(np.float32)
    
    return_list = []
    for query_box in new_data_box2d:
        for idx,key_box in enumerate(old_data_box2d):
            error = query_box - key_box
            if np.max(error)==0 and np.min(error==0):
                return_list.append(idx)
                break
    
    assert len(return_list)==len(new_data)
    return return_list


def update_old_data(old_data,new_data,index):
    
    for idx, n_d in enumerate(new_data):
        
        old_data[index[idx]] = n_d
        

    return old_data






if __name__=="__main__":
    

    parser = argparse.ArgumentParser(description="Convert VSRD Format into KITTI3D Dataset Configuration")
    parser.add_argument("--estimated_label", type=str, default="datasets/KITTI-360")
    parser.add_argument("--gt_labels",type=str, default="/data1/liu/VSRD_PP_Sync/det2d")
    parser.add_argument("--ratio",type=float,default=0.025)
    args = parser.parse_args()
    
    estimated_label = args.estimated_label
    gt_labels = args.gt_labels
    ratio = args.ratio
    
    adjustment_ratio = 0.75
    

    new_updated_label = estimated_label.replace("label_2","label_2_update_v1")
    os.makedirs(new_updated_label,exist_ok=True)

    cnt = 0
    for fname in tqdm(os.listdir(gt_labels)):

        gt_label_name = os.path.join(gt_labels,fname)
        est_label_name = gt_label_name.replace(gt_labels,estimated_label)
        
        # To Save
        saved_update_label_name = os.path.join(new_updated_label,fname)
        
        assert os.path.exists(gt_label_name)
        assert os.path.exists(est_label_name)
        
        gt_data = np.loadtxt(gt_label_name,dtype=str).reshape(-1,17)
        
        est_label_data = np.loadtxt(est_label_name,dtype=str).reshape(-1,16)
        
        gt_box2d =  gt_data[:,4:8].astype(np.float32)
        est_box2d = est_label_data[:,4:8].astype(np.float32)
        

        matched_index_gt, matched_index_internimage = bilateral_matching_linear_sum_assignment(boxes_A=gt_box2d,
                                                                                               boxes_B=est_box2d)
        
        
        new_gt_data  = gt_data[matched_index_gt]
        new_est_data = est_label_data[matched_index_internimage]
        
        
        if len(new_gt_data)==0:

            np.savetxt(saved_update_label_name,
                        est_label_data,
                        fmt='%s ' * 16,  
                        delimiter=' ' )
            
            
            continue
        dynamic_objects_signals = new_gt_data[:,-1].astype(np.float32)
        dynamic_objects_signal_bool=[a==1 for a in dynamic_objects_signals]
        
        if sum(dynamic_objects_signal_bool)==0:

            a = random.random()
            if a<ratio:
                cnt = cnt +1
                np.savetxt(saved_update_label_name,
                            est_label_data,
                            fmt='%s ' * 16,  
                            delimiter=' ' )
            else:
                np.savetxt(saved_update_label_name,
                            est_label_data,
                            fmt='%s ' * 16,  
                            delimiter=' ' )
            continue
        
        else:
            # This is the dynamic objects
            new_gt_data = new_gt_data[dynamic_objects_signal_bool]
            new_est_data = new_est_data[dynamic_objects_signal_bool]
            
            
            new_est_location = new_est_data[:,11:14].astype(np.float32)
            new_gt_location = new_gt_data[:,11:14].astype(np.float32)
            
            assert new_est_location.shape == new_gt_location.shape
            
            if np.max(np.abs(new_est_location-new_gt_location))>4:
                updated_est_location = adjustment_ratio* new_gt_location + (1-adjustment_ratio) * new_est_location
            else:
                updated_est_location = new_est_location
            
            
            new_est_data[:,11:14] = updated_est_location.astype(np.str_)
            
            return_list = found_the_index(new_data=new_est_data,old_data=est_label_data)
            
            updated_est_data = update_old_data(old_data=est_label_data,new_data=new_est_data,
                                               index=return_list)
            
            assert updated_est_data.shape == est_label_data.shape

            np.savetxt(saved_update_label_name,
                        updated_est_data,
                        fmt='%s ' * 16,  
                        delimiter=' ' )
            

                
            
            

             
        

        
    #     quit()
        

    
    # print(cnt/len(os.listdir(gt_labels)))
    # print("Ok")


        