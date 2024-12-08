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
    
    # filenames path
    
    sample_filename_root_folder = os.path.join(args.root_folder,"filenames/R50-N16-M128-B16/")
    
    
    sub_folder_name_list = ['2013_05_28_drive_0000_sync','2013_05_28_drive_0002_sync','2013_05_28_drive_0003_sync','2013_05_28_drive_0004_sync','2013_05_28_drive_0005_sync',
                            '2013_05_28_drive_0006_sync','2013_05_28_drive_0007_sync','2013_05_28_drive_0009_sync','2013_05_28_drive_0010_sync']
    
    
    saved_folder = os.path.join(args.root_folder,args.output_folder_name)
    os.makedirs(saved_folder,exist_ok=True)
    
    # All Training Contents
    total_saved_contents_list = []
    for idx, sub_folder_name in enumerate( sub_folder_name_list):
        sub_folder_name_full = os.path.join(sample_filename_root_folder,sub_folder_name)    
        sub_sequence_name = os.path.join(sub_folder_name_full,"sampled_image_filenames.txt")
        assert os.path.exists(sub_sequence_name)
        readed_contents = read_text_lines(sub_sequence_name)
        for sub_idx, content in enumerate(readed_contents):
            total_saved_contents_list.append(content)
    
    saved_all_contents_filename = os.path.join(saved_folder,"train_all_filenames.txt")
    
    with open(saved_all_contents_filename,'w') as f:
        for idx, line in enumerate(total_saved_contents_list):
            if idx!=len(total_saved_contents_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)
    
    
    # Split into 64 sub-set for traininig .
    NUM_SPLITS = args.num_splits
    saved_sub_nums_splits = os.path.join(saved_folder,"filename_split{}".format(NUM_SPLITS))
    os.makedirs(saved_sub_nums_splits,exist_ok=True)
    splited_lists = split_list(total_saved_contents_list,NUM_SPLITS)
    
    
    for idx, sub_list in enumerate(splited_lists):
        saved_sub_name = os.path.join(saved_sub_nums_splits,"sub_{}.txt".format(idx))
        save_contents(filename=saved_sub_name,contents_list=sub_list)
        
    
    print("Processed Done, filenames can be found in {}".format(saved_folder))
    
    
    print("-------------------------------------------------------------------")


    
    # filenames path    
    sample_filename_root_folder = os.path.join(args.root_folder,"est_dynamic_list")
    sub_folder_name_list = ['sync00','sync02','sync03','sync04','sync05',
                            'sync06','sync07','sync09','sync10']
    

    
    # All Training Contents
    total_saved_contents_list = []
    for idx, sub_folder_name in enumerate( sub_folder_name_list):
        sub_folder_name_full = os.path.join(sample_filename_root_folder,sub_folder_name)    
        sub_sequence_name = os.path.join(sub_folder_name_full,"dynamic_mask.txt")
        assert os.path.exists(sub_sequence_name)
        readed_contents = read_text_lines(sub_sequence_name)
        for sub_idx, content in enumerate(readed_contents):
            total_saved_contents_list.append(content)
    
    saved_all_contents_filename = os.path.join(saved_folder,"train_all_dyanmic_filenames.txt")
    
    with open(saved_all_contents_filename,'w') as f:
        for idx, line in enumerate(total_saved_contents_list):
            if idx!=len(total_saved_contents_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)

    # Split into 64 sub-set for traininig .
    NUM_SPLITS = args.num_splits
    saved_sub_nums_splits = os.path.join(saved_folder,"dynamic_split{}".format(NUM_SPLITS))
    os.makedirs(saved_sub_nums_splits,exist_ok=True)
    splited_lists = split_list(total_saved_contents_list,NUM_SPLITS)
    for idx, sub_list in enumerate(splited_lists):
        saved_sub_name = os.path.join(saved_sub_nums_splits,"sub_{}.txt".format(idx))
        save_contents(filename=saved_sub_name,contents_list=sub_list)
        
    
    print("Processed Done, dynamics can be found in {}".format(saved_folder))

    


        
    
        
        
    


