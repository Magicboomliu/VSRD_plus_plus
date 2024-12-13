import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from tqdm import tqdm

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def start_with_2013(string):
    return string[string.index("2013_"):]


if __name__=="__main__":
    
    # Step 1: First Process the New VSRDPP Split
    
    # ['train.txt', 'trainval.txt', 'val.txt', 'test.txt']
    data_root_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/"
    synced_root_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT/"
    original_ROI_LiDAR_root_path = "/media/zliu/data12/dataset/KITTI/KITTI360/ROI_LiDAR_internimage"

    trainval_filename_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT/ImageSets/trainval.txt"
    train_filename_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT/ImageSets/train.txt"
    val_filename_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT/ImageSets/val.txt"
    test_filename_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT/ImageSets/test.txt"
        
    trainval_sync_filename_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT/training/sync_file.txt"
    synced_root_path_training = os.path.join(synced_root_path,'training','ROI_LiDAR_InternImage')
    os.makedirs(synced_root_path_training,exist_ok=True)
    
    
    test_sync_filename_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT/testing/sync_file.txt"
    synced_root_path_testing = os.path.join(synced_root_path,'testing','ROI_LiDAR_InternImage')
    os.makedirs(synced_root_path_testing,exist_ok=True)
    
    
    
    trainval_filename_list = read_text_lines(trainval_filename_path)
    trainval_sync_filename_list = read_text_lines(trainval_sync_filename_path)
    assert len(trainval_filename_list) == len(trainval_sync_filename_list)
    
    train_filename_list = read_text_lines(train_filename_path)
    val_filename_list = read_text_lines(val_filename_path)
    test_filename_list = read_text_lines(test_filename_path)
    test_sync_filename_list = read_text_lines(test_sync_filename_path)
    
    
    # Create the CookBook for the test
    real_sync_test_dict = dict()
    for idx, line in enumerate(test_sync_filename_list):
        splits = line.split()
        real_image2_path = splits[0]
        synced_image2_path = splits[1]
        real_image2_path = start_with_2013(real_image2_path)
        synced_image2_path = os.path.basename(synced_image2_path)[:-4]
        real_sync_test_dict[synced_image2_path] = real_image2_path
    
    new_test_list = []
    for fname in tqdm(test_filename_list):
        real_image2_path = real_sync_test_dict[fname]
        original_ROI_LiDAR_path =  os.path.join(original_ROI_LiDAR_root_path,real_image2_path).replace(".png",".pkl")
        try:
            assert os.path.exists(original_ROI_LiDAR_path)
            new_test_list.append(fname)
        except:
            pass
    

    

    # Create the CookBook for trainval
    real_sync_dict = dict()
    for idx, line in enumerate(trainval_sync_filename_list):
        splits = line.split()
        real_image2_path = splits[0]
        synced_image2_path = splits[1]
        real_image2_path = start_with_2013(real_image2_path)
        synced_image2_path = os.path.basename(synced_image2_path)[:-4]
        real_sync_dict[synced_image2_path] = real_image2_path
        
    
    new_train_val_list = []
    new_train_list = []
    new_val_list = []
    missing_lines_list = []
    
    for fname in tqdm(trainval_filename_list):
        real_image2_path = real_sync_dict[fname]
        original_ROI_LiDAR_path =  os.path.join(original_ROI_LiDAR_root_path,real_image2_path).replace(".png",".pkl")
        try:
            assert os.path.exists(original_ROI_LiDAR_path)
            new_train_val_list.append(fname)
        except:
            missing_lines_list.append(original_ROI_LiDAR_path)
    
    
    for fname in tqdm(train_filename_list):
        if fname in new_train_val_list:
            new_train_list.append(fname)
        else:
            pass
        
    
    for fname in tqdm(val_filename_list):
        if fname in new_train_val_list:
            new_val_list.append(fname)
        else:
            pass



    new_WeakM3D_ImageSets = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT/ImageSets_For_WeakM3D/"
    os.makedirs(new_WeakM3D_ImageSets,exist_ok=True)
    
    new_trainval_save_filename = os.path.join(new_WeakM3D_ImageSets,'trainval.txt')
    with open(new_trainval_save_filename,'w') as f:
        for idx, line in enumerate(new_train_val_list):
            if idx!=len(new_train_val_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)
    
    new_train_save_filename = os.path.join(new_WeakM3D_ImageSets,'train.txt')
    with open(new_train_save_filename,'w') as f:
        for idx, line in enumerate(new_train_list):
            if idx!=len(new_train_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)       
    
    
    new_val_save_filename = os.path.join(new_WeakM3D_ImageSets,'val.txt')
    with open(new_val_save_filename,'w') as f:
        for idx, line in enumerate(new_val_list):
            if idx!=len(new_val_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)  
                

    new_test_save_filename = os.path.join(new_WeakM3D_ImageSets,'test.txt')
    with open(new_test_save_filename,'w') as f:
        for idx, line in enumerate(new_test_list):
            if idx!=len(new_test_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)    



    
    


