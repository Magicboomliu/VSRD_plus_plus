import torch
import os
import numpy as np
from tqdm import tqdm

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def save_into_txt(content_list,saved_name):
    with open(saved_name,'w') as f:
        for idx, line in enumerate(content_list):
            if idx!=len(content_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)


if __name__=="__main__":
    
    training_all_files_path = "/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/train_tsubame_filenames_ablation/train_all_filenames.txt"
    dynamic_all_files_path = "/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/train_tsubame_filenames_ablation/train_all_dynamic_mask.txt"
    
    train_all_files_data = read_text_lines(training_all_files_path)
    dynamic_all_files_data = read_text_lines(dynamic_all_files_path)
    
    updated_train_all_files_data = []
    updated_dynamic_all_files_data = []
    
    for line in tqdm(train_all_files_data):
        new_line = line.replace("/media/zliu/data12/dataset/VSRD_PP_Sync/","/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/")
        updated_train_all_files_data.append(new_line)
    
    for line in tqdm(dynamic_all_files_data):
        new_line = line.replace("/media/zliu/data12/dataset/VSRD_PP_Sync/","/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/")
        updated_dynamic_all_files_data.append(new_line)
    
    save_into_txt(updated_train_all_files_data,training_all_files_path)
    save_into_txt(updated_dynamic_all_files_data,dynamic_all_files_path)
    
    
    
    pass