import os
import numpy as np
from tqdm import tqdm
import argparse
import shutil
import re

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

def extract_from_2013_regex(string):
    """
    使用正则表达式从字符串中提取从 "2013" 开始的部分。
    如果未找到 "2013"，返回 None。
    """
    match = re.search(r"2013.*", string)
    return match.group(0) if match else None



if __name__=="__main__":
    '''
    Create the training and the validation from the train/val set
    Still use the sequence 10 as the testing set.
    '''

    # train and val split creation
    parser = argparse.ArgumentParser(description="Split the training and validation set")
    parser.add_argument("--root_dirname", type=str, default="/data1/liu/KITTI360_SoftLink/KITTI360_VSRDPP_V1")
    parser.add_argument("--sync_folder_path",type=str,default="")
    parser.add_argument("--pseudo_labels_dirname", type=str, default="")
    parser.add_argument("--output_dataset_path",type=str,default='daatset')
    parser.add_argument("--output_dataset_splits_filename_path",type=str,default='')
    args = parser.parse_args()
    
    

    root_dirname = args.root_dirname
    prediction_label_path = os.path.join(args.pseudo_labels_dirname,"predictions")
    gt_label_path_with_dynamic = os.path.join(args.pseudo_labels_dirname,"my_gts_with_dynamic")
    
    
    # training/validation/testing splits
    training_txt_filename = os.path.join(args.sync_folder_path,'ImageSets','train.txt')
    validation_txt_filename = os.path.join(args.sync_folder_path,'ImageSets','val.txt')
    training_validation_txt_filename = os.path.join(args.sync_folder_path,'ImageSets','trainval.txt')
    testing_txt_filename = os.path.join(args.sync_folder_path,'ImageSets','test.txt')
    
    assert os.path.exists(training_txt_filename)
    assert os.path.exists(validation_txt_filename)
    assert os.path.exists(training_validation_txt_filename)
    assert os.path.exists(testing_txt_filename)
    
    # sync for training and testing
    training_sync_fname = os.path.join(args.sync_folder_path,'training','sync_file.txt')
    testing_sync_fname = os.path.join(args.sync_folder_path,'testing','sync_file.txt')
    
    assert os.path.exists(training_sync_fname)
    assert os.path.exists(testing_sync_fname)
    
    
    # output folder
    output_folder = args.output_dataset_path
    output_folder_training = os.path.join(output_folder,'training')
    output_folder_testing = os.path.join(output_folder,'testing')

    
    output_folder_training_image_2_folder_path = os.path.join(output_folder_training,'image_2')
    output_folder_training_image_3_folder_path = os.path.join(output_folder_training,'image_3')
    output_folder_training_calib_folder_path = os.path.join(output_folder_training,"calib")
    output_folder_training_label_2_folder_path = os.path.join(output_folder_training,'label_2')
    output_folder_training_label_gt_folder_path = os.path.join(output_folder_training,'label_gt')
    

    output_folder_testing_image_2_folder_path = os.path.join(output_folder_testing,'image_2')
    output_folder_testing_image_3_folder_path = os.path.join(output_folder_testing,'image_3')
    output_folder_testing_calib_folder_path = os.path.join(output_folder_testing,"calib")
    output_folder_testing_label_2_folder_path = os.path.join(output_folder_testing,'label_2')
    output_folder_testing_label_gt_folder_path = os.path.join(output_folder_testing,'label_gt')
    
    os.makedirs(output_folder_training_image_2_folder_path,exist_ok=True)
    os.makedirs(output_folder_training_image_3_folder_path,exist_ok=True)
    os.makedirs(output_folder_training_calib_folder_path,exist_ok=True)
    os.makedirs(output_folder_training_label_2_folder_path,exist_ok=True)
    os.makedirs(output_folder_training_label_gt_folder_path,exist_ok=True)
    
    os.makedirs(output_folder_testing_image_2_folder_path,exist_ok=True)
    os.makedirs(output_folder_testing_image_3_folder_path,exist_ok=True)
    os.makedirs(output_folder_testing_calib_folder_path,exist_ok=True)
    os.makedirs(output_folder_testing_label_2_folder_path,exist_ok=True)
    os.makedirs(output_folder_testing_label_gt_folder_path,exist_ok=True)
    

    # for training set sync
    sync_train_contents_lines = read_text_lines(filepath=training_sync_fname)
    sync_test_contents_lines = read_text_lines(filepath=testing_sync_fname)
    
    
    sync_training_list = []
    for line in tqdm(sync_train_contents_lines):
        splits = line.strip().split()
        source_image_2,training_sync_target_image_2,source_image_3,training_sync_target_image_3,gt_label_path,training_sync_target_label_gt = splits
        
        old_root_dirname = source_image_2[:-len(extract_from_2013_regex(source_image_2))]
        
        # Image 2
        current_source_image_2 = source_image_2.replace(old_root_dirname,os.path.join(root_dirname,'data_2d_raw/'))
        current_syned_target_image_2 = os.path.join(output_folder_training_image_2_folder_path,os.path.basename(training_sync_target_image_2))
        assert os.path.exists(current_source_image_2)

        # Image 3
        current_source_image_3 = current_source_image_2.replace("image_00","image_01")
        current_syned_target_image_3 = os.path.join(output_folder_training_image_3_folder_path,os.path.basename(training_sync_target_image_2))
        
        assert os.path.exists(current_source_image_3)
        
        # Label 2
        current_source_label_2 =  os.path.join(prediction_label_path,extract_from_2013_regex(source_image_2)).replace(".png",".txt")
        current_syned_target_label_2 = os.path.join(output_folder_training_label_2_folder_path,os.path.basename(training_sync_target_image_2)).replace(".png",".txt")
        assert os.path.exists(current_source_label_2)
        
        # Label GT
        current_source_label_gt =  os.path.join(gt_label_path_with_dynamic,extract_from_2013_regex(source_image_2)).replace(".png",".txt")
        current_syned_target_label_gt = os.path.join(output_folder_training_label_gt_folder_path,os.path.basename(training_sync_target_image_2)).replace(".png",".txt")
        assert os.path.exists(current_source_label_gt)
        
        
        # Label Calib
        current_cam_calib = os.path.join(root_dirname,"cam_calib.txt")
        current_syned_target_cam_calib = os.path.join(output_folder_training_calib_folder_path,os.path.basename(training_sync_target_image_2)).replace(".png",".txt")        
        assert os.path.exists(current_cam_calib)

        
        # soft-link image 2
        if not os.path.exists(current_syned_target_image_2):
            os.system("ln -s {} {}".format(current_source_image_2,current_syned_target_image_2))
        
        # soft-link image 3
        if not os.path.exists(current_syned_target_image_3):
            os.system("ln -s {} {}".format(current_source_image_3,current_syned_target_image_3))
        
        # soft-link label 2 
        if not os.path.exists(current_syned_target_label_2):
            shutil.copy(current_source_label_2,current_syned_target_label_2)
        
        # soft-link label gt
        if not os.path.exists(current_syned_target_label_gt):
            shutil.copy(current_source_label_gt,current_syned_target_label_gt)
        
        # Soft-Link Calib
        if not os.path.exists(current_syned_target_cam_calib):
            shutil.copy(current_cam_calib,current_syned_target_cam_calib)
            
            
        # compose the sync list for training
        current_line = current_source_image_2  + " " + current_syned_target_image_2 + " " + current_source_image_3 + " " + current_syned_target_image_3 + " " + current_source_label_gt + " " + current_syned_target_label_gt
        sync_training_list.append(current_line)
    


    saved_training_sync_name = os.path.join(output_folder_training,'sync_file.txt')
    with open(saved_training_sync_name,'w') as f:
        for idx, line in enumerate(sync_training_list):
            if idx!=len(sync_training_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)
    
    

    # For the Testing Sync
    sync_testing_list = []
    for line in tqdm(sync_test_contents_lines):
        splits = line.strip().split()
        source_image_2,training_sync_target_image_2,source_image_3,training_sync_target_image_3,gt_label_path,training_sync_target_label_gt = splits
        
        old_root_dirname = source_image_2[:-len(extract_from_2013_regex(source_image_2))]
        
        # Image 2
        current_source_image_2 = source_image_2.replace(old_root_dirname,os.path.join(root_dirname,'data_2d_raw/'))
        current_syned_target_image_2 = os.path.join(output_folder_testing_image_2_folder_path,os.path.basename(training_sync_target_image_2))
        assert os.path.exists(current_source_image_2)

        # Image 3
        current_source_image_3 = current_source_image_2.replace("image_00","image_01")
        current_syned_target_image_3 = os.path.join(output_folder_testing_image_3_folder_path,os.path.basename(training_sync_target_image_2))
        
        assert os.path.exists(current_source_image_3)
        
        # Label 2
        current_source_label_2 =  os.path.join(prediction_label_path,extract_from_2013_regex(source_image_2)).replace(".png",".txt")
        current_syned_target_label_2 = os.path.join(output_folder_testing_label_2_folder_path,os.path.basename(training_sync_target_image_2)).replace(".png",".txt")
        assert os.path.exists(current_source_label_2)
        
        
        # Label GT
        current_source_label_gt =  os.path.join(gt_label_path_with_dynamic,extract_from_2013_regex(source_image_2)).replace(".png",".txt")
        current_syned_target_label_gt = os.path.join(output_folder_testing_label_gt_folder_path,os.path.basename(training_sync_target_image_2)).replace(".png",".txt")
        assert os.path.exists(current_source_label_gt)
        
        
        # Label Calib
        current_cam_calib = os.path.join(root_dirname,"cam_calib.txt")
        current_syned_target_cam_calib = os.path.join(output_folder_testing_calib_folder_path,os.path.basename(training_sync_target_image_2)).replace(".png",".txt")        
        assert os.path.exists(current_cam_calib)

        
        # soft-link image 2
        if not os.path.exists(current_syned_target_image_2):
            os.system("ln -s {} {}".format(current_source_image_2,current_syned_target_image_2))
        
        # soft-link image 3
        if not os.path.exists(current_syned_target_image_3):
            os.system("ln -s {} {}".format(current_source_image_3,current_syned_target_image_3))
        
        # soft-link label 2 
        if not os.path.exists(current_syned_target_label_2):
            shutil.copy(current_source_label_2,current_syned_target_label_2)
        
        # soft-link label gt
        if not os.path.exists(current_syned_target_label_gt):
            shutil.copy(current_source_label_gt,current_syned_target_label_gt)
        
        # Soft-Link Calib
        if not os.path.exists(current_syned_target_cam_calib):
            shutil.copy(current_cam_calib,current_syned_target_cam_calib)


        # compose the sync list for training
        current_line = current_source_image_2  + " " + current_syned_target_image_2 + " " + current_source_image_3 + " " + current_syned_target_image_3 + " " + current_source_label_gt + " " + current_syned_target_label_gt
        sync_testing_list.append(current_line)
            



    saved_testing_sync_name = os.path.join(output_folder_testing,'sync_file.txt')
    with open(saved_testing_sync_name,'w') as f:
        for idx, line in enumerate(sync_testing_list):
            if idx!=len(sync_testing_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)


    # Create the New ImageSets
    new_ImageSets = os.path.join(args.output_dataset_splits_filename_path,"ImageSets")
    source_ImageSets = os.path.join(args.sync_folder_path,'ImageSets')
    
    if not os.path.exists(new_ImageSets):
        os.system("cp -r {} {}".format(source_ImageSets,new_ImageSets))





    
     
    
    
    
    
    

    
    
    
