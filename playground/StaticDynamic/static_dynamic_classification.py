import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines



if __name__=="__main__":
    
    root_folder = "/media/zliu/data12/dataset/VSRD_PP_Sync/"
    
    estimated_velocites_path = "/media/zliu/data12/TPAMI_Results/EST_VELO/Estimated_Velocites/"
    assert os.path.exists(estimated_velocites_path)
        
    sync_files_path = "/media/zliu/data12/TPAMI_Results/TPAMI_Ablations/Velocity_Project_and_Sil_Loss/Sync_Files/"
    imageset_path = os.path.join(sync_files_path,"ImageSets")
    training_path = os.path.join(sync_files_path,"training")
    testing_path = os.path.join(sync_files_path,"testing")
    
    assert os.path.exists(imageset_path)
    assert os.path.exists(training_path)
    assert os.path.exists(testing_path)
    
    
    # for training files
    training_sync_files = os.path.join(training_path,"sync_file.txt")
    testing_sync_files = os.path.join(testing_path,"sync_file.txt")
    
    
    training_sync_files_contents = read_text_lines(training_sync_files)
    testing_sync_files_contents = read_text_lines(testing_sync_files)
    
    
    training_list_fname = "/media/zliu/data12/TPAMI_Results/TPAMI_Ablations/Velocity_Project_and_Sil_Loss/Sync_Files/ImageSets/train.txt"

    training_items = read_text_lines(training_list_fname)
    
    considered_images_list = []
    missed_velocities_list = []
    
    for idx,sample in enumerate(training_sync_files_contents):
        splits = sample.split()
        fname = splits[0]
        fname = fname.replace("/data3/","/media/zliu/data12/dataset/")
        assert os.path.exists(fname)
        synced_files = splits[1]
        fname_basename = os.path.basename(synced_files)[:-4]
        
        if fname_basename in training_items:
            considered_images_list.append(fname)

    for fname in tqdm(considered_images_list):

        saved_velocities_fname = fname.replace(root_folder,estimated_velocites_path)
        saved_velocities_fname = saved_velocities_fname.replace(".png",".pkl")

        try:
            assert os.path.exists(saved_velocities_fname)
        except:
            print(saved_velocities_fname)
            missed_velocities_list.append(saved_velocities_fname)
    


        


    
    
    
    
    
    
    