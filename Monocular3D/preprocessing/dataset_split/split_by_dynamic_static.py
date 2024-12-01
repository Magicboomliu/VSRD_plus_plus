import os
import numpy as np
from tqdm import tqdm
import argparse

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def starts_with_2013(list):
    for lst in list:
        if lst.startswith('2013'):
            return lst
    return None


def saved_into_txt(saved_name,content_list):
    with open(saved_name,'w') as f:
        for idx, content in enumerate(content_list):
            if idx!=len(content_list)-1:
                f.writelines(content+"\n")
            else:
                f.writelines(content)


if __name__ =="__main__":
    
    # train and val split creation
    parser = argparse.ArgumentParser(description="Split the training and validation set")
    parser.add_argument("--root_dirname", type=str, default="/data1/liu/KITTI360_SoftLink/KITTI360_VSRDPP_V1")
    parser.add_argument("--training_split",type=str,default="dataset")
    parser.add_argument("--valiation_split",type=str,default="dataset")
    parser.add_argument("--testing_split",type=str,default='daatset')
    parser.add_argument("--output_folder",type=str,default='daatset')


    args = parser.parse_args()

    sync_filepath_trainval = os.path.join(args.root_dirname,'training','sync_file.txt')
    sync_filepath_testing = os.path.join(args.root_dirname,'testing','sync_file.txt')
    train_list_splits = args.training_split
    val_list_splits = args.valiation_split
    test_list_splits = args.testing_split
    saved_location = args.output_folder
    os.makedirs(saved_location,exist_ok=True)
    
    sequence_dict = {"00":"2013_05_28_drive_0000_sync",
                     '02':"2013_05_28_drive_0002_sync",
                     '03':"2013_05_28_drive_0003_sync",
                     '04':"2013_05_28_drive_0004_sync",
                     '06':"2013_05_28_drive_0006_sync",
                     '05':"2013_05_28_drive_0005_sync",
                     '07': "2013_05_28_drive_0007_sync",
                     '09':"2013_05_28_drive_0009_sync",
                     '10':"2013_05_28_drive_0010_sync"
                     }
    
    train_seq_list = []
    val_seq_list = []
    trainval_seq_list = []
    test_seq_list = []
    
    
    for seq in train_list_splits.split(","):
        train_seq_list.append(sequence_dict[seq])
        
    for seq in val_list_splits.split(","):
        val_seq_list.append(sequence_dict[seq])
        trainval_seq_list = train_seq_list + val_seq_list
    
    for seq in test_list_splits.split(","):
        test_seq_list.append(sequence_dict[seq])
      
    
    train_contents = []
    val_contents = []
    train_val_contents = []
    
    # training contents and val contents creation
    contents = read_text_lines(sync_filepath_trainval)
    for line in tqdm(contents):
        source_image_filename_left,saved_image_left, label_est_name_abs,saved_label_est,label_2_name_abs, saved_label_2= line.split(" ")
        current_seq = starts_with_2013(source_image_filename_left.split("/"))
        basename = os.path.basename(saved_image_left)[:-4]
        
        if current_seq in train_seq_list:
            train_contents.append(basename)
        elif current_seq in val_seq_list:
            val_contents.append(basename)
        elif current_seq in test_seq_list:
            raise NotImplementedError
        else:
            raise NotImplementedError
        
    train_val_contents = train_contents + val_contents
    # testing contents creation
    test_contents = []
    # training contents and val contents creation
    contents = read_text_lines(sync_filepath_testing)
    for line in tqdm(contents):
        source_image_filename_left,saved_image_left, label_est_name_abs,saved_label_est,label_2_name_abs, saved_label_2= line.split(" ")
        current_seq = starts_with_2013(source_image_filename_left.split("/"))
        basename = os.path.basename(saved_image_left)[:-4]
        
        if current_seq in test_seq_list:
            test_contents.append(basename)
        else:
            raise NotImplementedError
        
        
    
    saved_training_path = os.path.join(saved_location,'train.txt')
    saved_valiation_path = os.path.join(saved_location,'val.txt')
    saved_trainval_path = os.path.join(saved_location,'trainval.txt')
    saved_test_path = os.path.join(saved_location,'test.txt')
    
    saved_into_txt(saved_name=saved_training_path,
                   content_list=train_contents)
    
    saved_into_txt(saved_name=saved_valiation_path,
                   content_list=val_contents)
    saved_into_txt(saved_name=saved_trainval_path,
                   content_list=train_val_contents)
    saved_into_txt(saved_name=saved_test_path,
                   content_list=test_contents)
    
    

    
    
    
        

    