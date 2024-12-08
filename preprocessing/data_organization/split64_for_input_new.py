import os
import sys
import re
import numpy as np
import argparse


# read contents
def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines



def parse_args():
    
    parser = argparse.ArgumentParser(description="SceneFlow-Multi-Baseline Images")
    parser.add_argument(
        "--root_folder",
        type=str,
        default="/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync",
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    parser.add_argument(
        "--num_splits",
        type=int,
        default=64,
        help="Path to pretrained model or model identifier from huggingface.co/models.")


    parser.add_argument(
        "--output_folder_name",
        type=str,
        default="train_tsubame_filenames",
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    args = parser.parse_args()


    return args


def split_list(lst, n):
    # 每一部分的基本长度和需要多加一个元素的部分数量
    k, m = divmod(len(lst), n)
    # 划分列表，使前 m 个部分的长度为 k + 1，后面的部分长度为 k
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def save_contents(filename,contents_list):
    with open(filename,'w') as f:
        for idx, line in enumerate(contents_list):
            if idx!=len(contents_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)


if __name__=="__main__":
    
    args = parse_args()
    
    filename_path = "/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/train_tsubame_filenames/train_all_filenames_V1.txt"
    dynamic_list_path ="/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/train_tsubame_filenames/train_all_dyanmic_filenames_V1.txt"
    
    # filenames path
    saved_folder = os.path.join(args.root_folder,args.output_folder_name)
    os.makedirs(saved_folder,exist_ok=True)

    total_saved_contents_list = read_text_lines(filepath=filename_path)


    # Split into 64 sub-set for traininig .
    NUM_SPLITS = args.num_splits
    saved_sub_nums_splits = os.path.join(saved_folder,"filename_split{}".format(NUM_SPLITS))
    os.makedirs(saved_sub_nums_splits,exist_ok=True)
    splited_lists = split_list(total_saved_contents_list,NUM_SPLITS)
    
    
    for idx, sub_list in enumerate(splited_lists):
        saved_sub_name = os.path.join(saved_sub_nums_splits,"sub_{}.txt".format(idx))
        save_contents(filename=saved_sub_name,contents_list=sub_list)
        
    
    print("Processed Done, filenames can be found in {}".format(saved_folder))
    print("----------------------------------------------------------------------------------------------------")
    
    total_saved_contents_list = read_text_lines(filepath=dynamic_list_path)
    # Split into 64 sub-set for traininig .
    NUM_SPLITS = args.num_splits
    saved_sub_nums_splits = os.path.join(saved_folder,"dynamic_split{}".format(NUM_SPLITS))
    os.makedirs(saved_sub_nums_splits,exist_ok=True)
    splited_lists = split_list(total_saved_contents_list,NUM_SPLITS)
    for idx, sub_list in enumerate(splited_lists):
        saved_sub_name = os.path.join(saved_sub_nums_splits,"sub_{}.txt".format(idx))
        save_contents(filename=saved_sub_name,contents_list=sub_list)
        
    
    print("Processed Done, dynamics can be found in {}".format(saved_folder))

    


        
    
        
        
