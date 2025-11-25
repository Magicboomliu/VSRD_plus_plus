import os
import sys
import re

def update_config_file(filepath, new_filenames, new_labels_path, new_config_path):
    # 读取文件内容
    with open(filepath, 'r') as file:
        content = file.read()
    
    # 使用正则表达式替换 _C.TRAIN.DATASET.FILENAMES、_C.TRAIN.DYNAMIC_LABELS_PATH 和 _C.TRAIN.CONFIG
    content = re.sub(
        r"_C\.TRAIN\.DATASET\.FILENAMES = \[.*?\]",
        f"_C.TRAIN.DATASET.FILENAMES = [\"{new_filenames}\"]",
        content
    )
    content = re.sub(
        r"_C\.TRAIN\.DYNAMIC_LABELS_PATH=.*",
        f"_C.TRAIN.DYNAMIC_LABELS_PATH=\"{new_labels_path}\"",
        content
    )
    content = re.sub(
        r"_C\.TRAIN\.CONFIG = \".*?\"",
        f"_C.TRAIN.CONFIG = \"{new_config_path}\"",
        content
    )
    
    # 将更新后的内容写回文件
    with open(filepath, 'w') as file:
        file.write(content)
    


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SceneFlow-Multi-Baseline Images")
    parser.add_argument(
        "--configs_root_path",
        type=str,
        default="/home/zliu/CVPR25_Detection/Submitted_Version/VSRD-V2/Optimized_Based/configs/SPLITS64/",
        help="")

    parser.add_argument(
        "--filenames_path",
        type=str,
        default="/media/zliu/data12/dataset/VSRD_PP_Sync/train_tsubame_filenames/filename_split64/",
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    parser.add_argument(
        "--dynamic_path",
        type=str,
        default="/media/zliu/data12/dataset/VSRD_PP_Sync/train_tsubame_filenames/dynamic_split64/",
        help="Path to pretrained model or model identifier from huggingface.co/models.")



    # get the local rank
    args = parser.parse_args()


    return args




if __name__=="__main__":
    
    args = parse_args()
    
    
    configs_root_path = args.configs_root_path
    filenames_list_path = args.filenames_path
    dynamic_list_path = args.dynamic_path
    
    
    original_file = "train_config_sub_00.py"
    original_file = os.path.join(configs_root_path,original_file)
    
    N_SPLITS = 64
    for i in range(N_SPLITS):
        suffix = f"{i:02}"
        new_file = f"train_config_sub_{suffix}.py"
        new_file = os.path.join(configs_root_path,new_file)
        
        new_config_filenames = os.path.join(filenames_list_path,"sub_{}.txt".format(i))
        new_dynamic_filenames = os.path.join(dynamic_list_path,"sub_{}.txt".format(i))
        new_config_root_path = args.configs_root_path
        
        update_config_file(filepath=new_file,new_filenames= new_config_filenames,
                           new_labels_path=new_dynamic_filenames,
                           new_config_path=new_config_root_path)