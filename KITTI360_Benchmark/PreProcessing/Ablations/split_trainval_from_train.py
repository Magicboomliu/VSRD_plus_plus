import os
import numpy as np
import sys
import shutil
from tqdm import tqdm

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


if __name__=="__main__":
    
    psuedo_label_name = "/media/zliu/data12/TPAMI_Results/Stage1_Expermental_Results/Synced_For_Evaluations/Splits_2024/VSRD_Vanilla/"
    
    stage1_evaluation = os.path.join(psuedo_label_name,'stage1_eval')
    os.makedirs(stage1_evaluation,exist_ok=True)
    os.makedirs(os.path.join(stage1_evaluation,'label_2'),exist_ok=True)
    os.makedirs(os.path.join(stage1_evaluation,'label_gt'),exist_ok=True)
    
    trainfiles_path = os.path.join(psuedo_label_name,"ImageSets/train.txt")

    
    trainlines = read_text_lines(trainfiles_path)
    
    
    training_files_est = os.path.join(psuedo_label_name,'training/label_2')

    for fname in tqdm(sorted(os.listdir(training_files_est))):
        
        fname_abs_est = os.path.join(training_files_est,fname)
        fname_abs_gt =fname_abs_est.replace("label_2","label_gt")
        assert fname_abs_gt!=fname_abs_est
        assert os.path.exists(fname_abs_est)
        assert os.path.exists(fname_abs_gt)
        
        if fname[:-4] in trainlines:
            
            copyed_fname_abs_est = fname_abs_est.replace("training",'stage1_eval')
            copyed_fname_abs_gt = fname_abs_gt.replace("training",'stage1_eval')
            
            shutil.copy(src=fname_abs_est,dst=copyed_fname_abs_est)
            shutil.copy(src=fname_abs_gt,dst=copyed_fname_abs_gt)
