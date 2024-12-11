import torch
import numpy as np
import torch.nn.functional as F
import os



def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def start_with_data_2d(string):
    return string[string.index("data_2d_raw"):]



if __name__=="__main__":
    
    old_sequence_training_sync_file = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT/training/sync_file.txt"
    old_sequence_testing_sync_file = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT/testing/sync_file.txt"
    
    assert os.path.exists(old_sequence_training_sync_file)
    assert os.path.exists(old_sequence_testing_sync_file)
    
    old_trainval_contents = read_text_lines(old_sequence_training_sync_file)
    
    old_test_contents = read_text_lines(old_sequence_testing_sync_file)
    

    all_filenames_list = []
    
    for content in old_trainval_contents:
        filename = content.split()[0]
        sync_label = content.split()[1]
        filename = start_with_data_2d(filename)
        
        all_filenames_list.append(filename+" "+sync_label)
    
    for content in old_test_contents:
        filename = content.split()[0]
        sync_label = content.split()[1]
        filename = start_with_data_2d(filename)
        
        all_filenames_list.append(filename+" "+sync_label)
    
    
    with open("all_filenames.txt",'w') as f:
        for idx, line in enumerate(all_filenames_list):
            if idx!=len(all_filenames_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)

