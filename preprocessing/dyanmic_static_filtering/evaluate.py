import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import re


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def judge_dynamic_by_threshold(value,threshold):
    if value>threshold:
        return 1.0
    else:
        return 0.0
    

def filter_and_sort(A, B):
    """
    根据 B 中的 ID，过滤掉 A 中不存在于 B 的 ID，保留顺序一致。
    
    :param A: 包含 (ID, value) 的列表
    :param B: 包含 (ID, value) 的列表
    :return: 过滤并按照 B 顺序排列的 A 中的项
    """
    # 创建字典来快速查找 A 中的值
    A_dict = {id_.strip(): value for id_, value in A}

    # 保留在 B 中出现的项，并且按照 B 的顺序重新排列
    result = [(id_.strip(), A_dict[id_.strip()]) for id_, _ in B if id_.strip() in A_dict]

    return result
    
    

def calculate_accuracy_recall(TP, TN, FP, FN):
    '''
    T is the dynamic
    N is the static
    
    '''
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    
    static_rate = TN/(TN+FP)
    return accuracy, recall,static_rate,(TP+FN),(TN+FP)



if __name__=="__main__":
    
    gt_dynamic_mask = "/media/zliu/data12/dataset/KITTI/VSRD_Format/dynamic_static/2013_05_28_drive_0003_sync_gt.txt"
    result_dynamic_mask = "/media/zliu/data12/dataset/KITTI/VSRD_Format/dynamic_static/2013_05_28_drive_0003_sync_est.txt"
    
    gt_contents = read_text_lines(gt_dynamic_mask)
    result_contents = read_text_lines(result_dynamic_mask)
    
    assert len(result_contents) == len(gt_contents)

    # pattern = r"(\d+,\s*\d+)\s+(/media[^\s]+)\s+([\d\.,\s]+)"
    
    pattern = r"([\d,\s]+)\s+(/media[^\s]+)\s+([\d\.,\s]+)"
    
    threshold = 1.1
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    
    
    for idx, line in enumerate(gt_contents):
        
        parts = line.strip()
        match_gt = re.match(pattern, line)
        match_estimated = re.match(pattern,result_contents[idx])

        numbers = match_gt.group(1)  # 匹配数字部分
        gt_floats = match_gt.group(3)   # 匹配浮点数部分
        
        
        est_numbers = match_estimated.group(1)
        est_floats = match_estimated.group(3)
        
        
        est_numbers = est_numbers.split(",")
        numbers = numbers.split(",")
        
        

        gt_floats = [float(x) for x in gt_floats.strip().split(',')]
        
        est_floats = [float(x) for x in est_floats.strip().split(',')]
        
        gt_dict = list(zip(numbers,gt_floats))
        est_dict = list(zip(est_numbers,est_floats))

        est_dict = filter_and_sort(A=est_dict,B=gt_dict)
        
        numbers_unzipped, est_floats_unzipped = zip(*est_dict)
        est_floats = [judge_dynamic_by_threshold(est,threshold) for est in est_floats_unzipped]
        
        
        for sub_id in range(len(est_floats)):
            est_values = est_floats[sub_id]
            gt_values = gt_floats[sub_id]
            
            if ((est_values==1) and (gt_values==1)):
                TP = TP +1
            if ((est_values==1) and (gt_values==0)):
                FP = FP +1

            if ((est_values==0) and (gt_values==0)):
                TN = TN +1
            if ((est_values==0) and (gt_values==1)):
                FN = FN +1
        
    
    
    acc,recall,static_rate,dynamic,static= calculate_accuracy_recall(TP=TP,TN=TN,FP=FP,FN=FN)     
        

    print(acc)
    print(recall)
    print(static_rate)
    print(dynamic)
    print(static)
        

    
    
    
    pass