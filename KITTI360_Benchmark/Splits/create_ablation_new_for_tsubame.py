import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

import numpy as np

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def cut_with_beginning_of_2013(string):
    return string[string.index("2013"):]

def cut_with_begining_of_data_2d_raw(string):
    return string[string.index("data_2d"):]



if __name__=="__main__":
    
    output_root_folder = "/media/zliu/data12/dataset/VSRD_PP_Sync/train_tsubame_ablations_filenames/"
    os.makedirs(output_root_folder,exist_ok=True)
    
    ablation_studies_sequence_name = "/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/ablation_studies_sequence_name.txt"
    
    
    completed_sampled_filenames = "/media/zliu/data12/dataset/VSRD_PP_Sync/train_tsubame_filenames/train_all_filenames.txt"
    completed_sampled_dynamic_filenames = "/media/zliu/data12/dataset/VSRD_PP_Sync/train_tsubame_filenames/train_all_dynamic_mask.txt"
    
    
    saved_ablation_sampled_filenames_path = os.path.join(output_root_folder,'train_all_filenames.txt')
    saved_ablation_sampled_dynamic_path = os.path.join(output_root_folder,'train_all_dynamic_mask.txt')
    
    
    
    completed_sampled_filenames_data = read_text_lines(completed_sampled_filenames)
    completed_sample_dynamic_filenames_data = read_text_lines(completed_sampled_dynamic_filenames)
    ablation_studies_filename_data = read_text_lines(ablation_studies_sequence_name)
    ablation_studies_filename_data = [os.path.dirname(cut_with_begining_of_data_2d_raw(line)) for line in ablation_studies_filename_data]
    
    
    new_sample_image_filename_list = []
    new_dynamic_mask_filename_list = []
    
    
    

    for idx, line in enumerate(completed_sampled_filenames_data):
        sampled_image_filename = line.split()[1]
        dynamic_mask_filename = completed_sample_dynamic_filenames_data[idx].split()[1]
        
        reference_sampled_fname_from_samplefilenames = cut_with_begining_of_data_2d_raw(sampled_image_filename)[:-4]
        reference_sampled_fname_from_dynamicmasks = cut_with_begining_of_data_2d_raw(dynamic_mask_filename)[:-4]
        assert reference_sampled_fname_from_samplefilenames == reference_sampled_fname_from_dynamicmasks
    
        if reference_sampled_fname_from_samplefilenames in ablation_studies_filename_data:
            new_sample_image_filename_list.append(completed_sampled_filenames_data[idx])
            new_dynamic_mask_filename_list.append(completed_sample_dynamic_filenames_data[idx])


    with open(saved_ablation_sampled_filenames_path,'w') as f:
        for idx, line in enumerate(new_sample_image_filename_list):
            if idx!=len(new_sample_image_filename_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)

    with open(saved_ablation_sampled_dynamic_path,'w') as f:
        for idx, line in enumerate(new_dynamic_mask_filename_list):
            if idx!=len(new_dynamic_mask_filename_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line+"\n")
    
    
    
    
    # print(len(new_sample_image_filename_list))
    # print(len(new_dynamic_mask_filename_list))
        

 


