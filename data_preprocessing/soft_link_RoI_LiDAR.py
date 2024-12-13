import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

from tqdm import tqdm

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def start_with_2013(string):
    return string[string.index("2013_"):]



if __name__=="__main__":
    
    data_root_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/"
    synced_root_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/"
    original_ROI_LiDAR_root_path = "/media/zliu/data12/dataset/KITTI/KITTI360/ROI_LiDAR_internimage"


    trainval_filename_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/ImageSets_For_WeakM3D/trainval.txt"
    train_filename_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/ImageSets_For_WeakM3D/train.txt"
    val_filename_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/ImageSets_For_WeakM3D/val.txt"
    test_filename_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/ImageSets_For_WeakM3D/test.txt"



    trainval_sync_filename_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/training/sync_file.txt"
    synced_root_path_training = os.path.join(synced_root_path,'training','ROI_LiDAR_InternImage')
    
    # training det2d
    synced_root_det2d_training = os.path.join(synced_root_path,'training','det2d')
    os.makedirs(synced_root_det2d_training,exist_ok=True)

    
    
    test_sync_filename_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/testing/sync_file.txt"
    synced_root_path_testing = os.path.join(synced_root_path,'testing','ROI_LiDAR_InternImage')
    os.makedirs(synced_root_path_testing,exist_ok=True)


    # training det2d
    synced_root_det2d_testing = os.path.join(synced_root_path,'testing','det2d')
    os.makedirs(synced_root_det2d_testing,exist_ok=True)
        


    trainval_filename_list = read_text_lines(trainval_filename_path)
    trainval_sync_filename_list = read_text_lines(trainval_sync_filename_path)
    train_filename_list = read_text_lines(train_filename_path)
    val_filename_list = read_text_lines(val_filename_path)
    assert len(trainval_filename_list)< len(trainval_sync_filename_list)
    

    
    test_filename_list = read_text_lines(test_filename_path)
    test_sync_filename_list = read_text_lines(test_sync_filename_path)
    assert len(test_filename_list) < len(test_sync_filename_list)


    # First Sync the Training and the Validation ROI Internimage
    
    # Create the CookBook for trainval
    real_sync_dict = dict()
    for idx, line in enumerate(trainval_sync_filename_list):
        splits = line.split()
        real_image2_path = splits[0]
        synced_image2_path = splits[1]
        real_image2_path = start_with_2013(real_image2_path)
        synced_image2_path = os.path.basename(synced_image2_path)[:-4]
        real_sync_dict[synced_image2_path] = real_image2_path

    for fname in tqdm(trainval_filename_list):
        real_image2_path = real_sync_dict[fname]
        original_ROI_LiDAR_path =  os.path.join(original_ROI_LiDAR_root_path,real_image2_path).replace(".png",".pkl")
        assert os.path.exists(original_ROI_LiDAR_path)
        
        synced_ROI_LiDAR_Path = os.path.join(synced_root_path_training,fname)+".pkl"
        
        if not os.path.exists(synced_ROI_LiDAR_Path):
            os.system("ln -s {} {}".format(original_ROI_LiDAR_path,synced_ROI_LiDAR_Path))
            
        # estimated liDAR
        original_det2d_training_path = os.path.join(data_root_path,"det2d/threshold03/",real_image2_path.replace(".png",".txt"))
        if os.path.exists(original_det2d_training_path):
            synced_det2d_fname_path = os.path.join(synced_root_det2d_training,fname)+".txt"
            if not os.path.exists(synced_det2d_fname_path):
                os.system("ln -s {} {}".format(original_det2d_training_path,synced_det2d_fname_path))
                
        
    
    #-----------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------#    
    
    # First Sync the Testing ROI Internimage

    real_sync_test_dict = dict()
    for idx, line in enumerate(test_sync_filename_list):
        splits = line.split()
        real_image2_path = splits[0]
        synced_image2_path = splits[1]
        real_image2_path = start_with_2013(real_image2_path)
        synced_image2_path = os.path.basename(synced_image2_path)[:-4]
        real_sync_test_dict[synced_image2_path] = real_image2_path

    cnt = 0
    for fname in tqdm(test_filename_list):
        real_image2_path = real_sync_test_dict[fname]
        original_ROI_LiDAR_path =  os.path.join(original_ROI_LiDAR_root_path,real_image2_path).replace(".png",".pkl")
        assert os.path.exists(original_ROI_LiDAR_path)
        
        synced_ROI_LiDAR_Path = os.path.join(synced_root_path_testing,fname)+".pkl"

        if not os.path.exists(synced_ROI_LiDAR_Path):
            os.system("ln -s {} {}".format(original_ROI_LiDAR_path,synced_ROI_LiDAR_Path))

        # estimated liDAR
        original_det2d_testing_path = os.path.join(data_root_path,"det2d/threshold03/",real_image2_path.replace(".png",".txt"))
        if os.path.exists(original_det2d_testing_path):
            cnt = cnt + 1
            synced_det2d_fname_path = os.path.join(synced_root_det2d_testing,fname)+".txt"
            if not os.path.exists(synced_det2d_fname_path):
                os.system("ln -s {} {}".format(original_det2d_testing_path,synced_det2d_fname_path))

    print(cnt)
