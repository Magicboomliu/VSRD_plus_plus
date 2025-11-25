import os
import sys
from tqdm import tqdm
import re

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SceneFlow-Multi-Baseline Images")
    parser.add_argument(
        "--gt_original_root_folder",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    parser.add_argument(
        "--changed_root_folder_root",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    # get the local rank
    args = parser.parse_args()


    return args


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def find_nearest(input_list, neigh_nums=16, threshold=25):
    # 生成等差数列候选项，确保范围在 [-threshold, threshold]
    target_difference = 1  # 可以根据需求调整公差
    arithmetic_sequence_candidates = list(range(-threshold, threshold, target_difference))

    # 过滤出在 input_list 中的有效候选项
    valid_candidates = [num for num in arithmetic_sequence_candidates if num in input_list]

    # 如果候选项不足指定的数目，直接返回全部
    if len(valid_candidates) <= neigh_nums:
        return sorted(valid_candidates)
    
    # 否则，均匀采样16个元素
    step = len(valid_candidates) / neigh_nums
    sampled_candidates = [valid_candidates[int(i * step)] for i in range(neigh_nums)]

    return sorted(sampled_candidates)


def narrow_the_relative_frame_idx(input_list):
    
    input_list = sorted(input_list)
    if len(input_list)<17:
        return input_list
    else:
        return find_nearest(input_list)



def Process_the_Group_Data(original_sample_txt_path,
                           
                           reference_origina_root_folder_in_txt,
                           changed_root_folder_name,
                           threshold=35):
    
    contents = read_text_lines(original_sample_txt_path)

    new_sample_txt_contents_list = []
    
    for content in contents:
        splits = content.split(" ")
        instance_name = splits[0]
        changed_source_name = splits[1].replace(reference_origina_root_folder_in_txt,changed_root_folder_name)        
        source_frame_idx = splits[2]
        source_frame_idx_list = source_frame_idx.split(",")
        source_frame_idx_list = [int(item) for item in source_frame_idx_list]
        min_relative_frame_idx = min(source_frame_idx_list)
        max_relative_frame_idx = max(source_frame_idx_list)
        
        
        if ((-1*min_relative_frame_idx)>threshold) or (max_relative_frame_idx >threshold):
            source_frame_idx_list = narrow_the_relative_frame_idx(source_frame_idx_list)
        
        source_frame_idx_str = ','.join(map(str, source_frame_idx_list))
        
        new_content = instance_name+" "+changed_source_name+" "+source_frame_idx_str
        new_sample_txt_contents_list.append(new_content)
    
    
    return new_sample_txt_contents_list




def update_dynamic_mask(initial_path,update_path,temp1,temp2):
    
    
    dyanmic_initial_list = read_text_lines(initial_path)
    
    
    saved_contents = []
    for line in dyanmic_initial_list:
        new_line = line.replace(temp1,temp2)
        saved_contents.append(new_line)
    
    with open(update_path,'w') as f:
        for idx, line in enumerate(saved_contents):
            if idx!=len(saved_contents)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)





if __name__=="__main__":
    
    args = parse_args()
    
    original_root_folder_in_txt = "/media/zliu/data12/dataset/KITTI/VSRD_Format"
    
    dynamic_old_template_in_txt = "/data1/liu/VSRD_PP_Sync"

    gt_original_root_folder = args.gt_original_root_folder
    changed_root_folder_root = args.changed_root_folder_root
    



    # for sample groups
    root_folder = "{}/filenames/R50-N16-M128-B16/".format(gt_original_root_folder)
    changed_root_folder = "{}/filenames/R50-N16-M128-B16".format(changed_root_folder_root)
    
    for sub_folder in tqdm(sorted(os.listdir(root_folder))):
        
        current_sub_folder_name = os.path.join(root_folder,sub_folder)
        current_target_folder_name = os.path.join(changed_root_folder,sub_folder)
        os.makedirs(current_target_folder_name,exist_ok=True)
        source_group_name = os.path.join(current_sub_folder_name,"sampled_image_filenames.txt")
        saved_target_group_name = os.path.join(current_target_folder_name,"sampled_image_filenames.txt")
        assert os.path.exists(source_group_name)
        
        new_sample_txt_contents_list = Process_the_Group_Data(source_group_name,
                               reference_origina_root_folder_in_txt = original_root_folder_in_txt,
                               changed_root_folder_name=changed_root_folder_root,
                               threshold=25)
        
        with open(saved_target_group_name,'w') as f:
            for idx, line in enumerate(new_sample_txt_contents_list):
                if idx!=len(new_sample_txt_contents_list)-1:
                    f.writelines(line+"\n")
                else:
                    f.writelines(line)
        
    # for dynamic mask

    root_folder = "{}/est_dynamic_list/".format(gt_original_root_folder)
    changed_root_folder = "{}/est_dynamic_list/".format(changed_root_folder_root)
    

    for sub_folder in tqdm(sorted(os.listdir(root_folder))):
        current_sub_folder_name = os.path.join(root_folder,sub_folder)
        original_dyanmic_mask_path = os.path.join(current_sub_folder_name,"dynamic_mask.txt")

        assert os.path.exists(original_dyanmic_mask_path)
        
        saved_name_folder = os.path.join(changed_root_folder,sub_folder)
        os.makedirs(saved_name_folder,exist_ok=True)
        saved_name = os.path.join(saved_name_folder,"dynamic_mask.txt")
        

        update_dynamic_mask(initial_path=original_dyanmic_mask_path,
                            update_path=saved_name,
                            temp1=dynamic_old_template_in_txt,temp2=changed_root_folder_root)
        
    
    
    
        
    print("All Processed is Finished! Done")
    
    
