import torch
import os
import numpy as np
import sys
import shutil


if __name__=="__main__":
    
    est_folder = "/home/zliu/TPAMI25/Mono3Ds/MonoDETR/output/vsrd24_splits/Autolabels/monodetr/outputs/data/"
    gt_folder = "/data3/PAMI_Datasets/Replaced_With_GT_Boxes/Autolabels/VSRD24_Splits/testing/label_gt"
    gt_folder2 = "/data3/PAMI_Datasets/Replaced_With_GT_Boxes/Autolabels/VSRD24_Splits/testing/label_gt2"

    os.makedirs(gt_folder2,exist_ok=True)
    estimated_folder = sorted(os.listdir(est_folder))
    
    new_gt_list = []
    

    for fname in sorted(os.listdir(gt_folder)):

        if fname in estimated_folder:
            print("Yes!")
            source_label_gt = os.path.join(gt_folder,fname)
            assert os.path.exists(source_label_gt)
            
            target_label_gt = os.path.join(gt_folder2,fname)
            shutil.copy(source_label_gt,target_label_gt) 
            
            
