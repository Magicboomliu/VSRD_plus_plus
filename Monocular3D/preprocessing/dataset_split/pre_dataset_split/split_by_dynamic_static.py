import os
import numpy as np
from tqdm import tqdm
import argparse
import shutil
import random
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
    parser.add_argument("--pseudo_labels_dirname", type=str, default="")
    parser.add_argument("--dynamic_static_ratio",type=float,default=0.25)
    parser.add_argument("--trainval_ratio",type=float,default=0.5)
    parser.add_argument("--testing_split",type=str,default='daatset')
    parser.add_argument("--output_dataset_path",type=str,default='daatset')
    parser.add_argument("--output_dataset_splits_filename_path",type=str,default='')
    args = parser.parse_args()
    

    root_dirname = args.root_dirname
    prediction_label_path = os.path.join(args.pseudo_labels_dirname,"predictions")
    gt_label_path_with_dynamic = os.path.join(args.pseudo_labels_dirname,"my_gts_with_dynamic")
    dynamic_static_ratio = args.dynamic_static_ratio
    testing_split = (args.testing_split).split(",")
    
    trainval_ratio = args.trainval_ratio
    
    
    
    # Split all the training splits for training.
    output_split_folder_name  = args.output_dataset_splits_filename_path
    output_split_folder_name = os.path.join(output_split_folder_name,"ImageSets")
    os.makedirs(output_split_folder_name,exist_ok=True)
    training_splits_filename_path_sync = os.path.join(output_split_folder_name,'train.txt')
    validation_splits_filename_path_sync = os.path.join(output_split_folder_name,'val.txt')
    training_validation_splits_filename_path_sync = os.path.join(output_split_folder_name,'trainval.txt')
    testing_splits_filename_path_sync = os.path.join(output_split_folder_name,'test.txt')
    training_filename_splits_ImageSets_list = []
    validation_filename_splits_ImageSets_list = []
    trainval_filename_splits_ImageSets_list = []
    testing_filename_splits_ImageSets_list = []
    

    
    # output folder
    output_folder = args.output_dataset_path
    output_folder_training = os.path.join(output_folder,'training')
    output_folder_testing = os.path.join(output_folder,'testing')
    All_Sample_Sequenes = ['00','02','03','04','05','06','07','09'] # 8 kinds
    All_Sample_Sequenes = sorted(All_Sample_Sequenes)
    evaluation_sequence = ['10'] # 10 testing
    

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

    
    sequence_dict = {"00":"2013_05_28_drive_0000_sync",
                     "02":"2013_05_28_drive_0002_sync",
                     "03":"2013_05_28_drive_0003_sync",
                     "04":"2013_05_28_drive_0004_sync",
                     "05":"2013_05_28_drive_0005_sync",
                     "06":"2013_05_28_drive_0006_sync",
                     "07":"2013_05_28_drive_0007_sync",
                     "09":"2013_05_28_drive_0009_sync",
                     "10":"2013_05_28_drive_0010_sync"}
    
    
    dynamic_instances_list = []
    staic_instance_list = []
    # get all the training/validation/testing splits
    for seq in tqdm(All_Sample_Sequenes):
        sequence_name = sequence_dict[seq]
        # Ground Truth Dynamic Label Path
        label_gt_with_dynamic_sub_folder = os.path.join(gt_label_path_with_dynamic,sequence_name,"image_00/data_rect/")
        assert os.path.exists(label_gt_with_dynamic_sub_folder)
        
        for label_name in sorted(os.listdir(label_gt_with_dynamic_sub_folder)):
            label_name_completed = os.path.join(label_gt_with_dynamic_sub_folder,label_name)
            
            label_data_gt_completed = np.loadtxt(label_name_completed,dtype=str).reshape(-1,17)
            dynamic_static_label = [float(item) for item in label_data_gt_completed[:,-1]]
            if 1 in dynamic_static_label:
                dynamic_instances_list.append(label_name_completed)
            else:
                staic_instance_list.append(label_name_completed)
    
    # random sample from the static instance to make the dynamic:staic = trainval_ratio
    total_training_instances_nums = int(len(dynamic_instances_list) * 1.0 / dynamic_static_ratio)
    training_rest_static_instance_nums = total_training_instances_nums - len(dynamic_instances_list)
    sampled_training_instances_from_the_rest = random.choices(staic_instance_list, k=training_rest_static_instance_nums)
    training_instances_all = sampled_training_instances_from_the_rest + dynamic_instances_list # training 
    
    
    # validation splits
    rest_static_instance_nums_all = []
    for fname in staic_instance_list:
        if fname in sampled_training_instances_from_the_rest:
            pass
        else:
            rest_static_instance_nums_all.append(fname)
    sampled_rest_static_instances_nums =  int(len(training_instances_all)/trainval_ratio) - len(training_instances_all)
    sampled_validation_instances_from_the_test = random.choices(rest_static_instance_nums_all, 
                                                                k=sampled_rest_static_instances_nums)  # validation
    validation_instances_all = sampled_validation_instances_from_the_test
    
    
    
    # train and validation splits
    trainval_instances_all = training_instances_all + validation_instances_all
    
    
    # evaluation splits
    evaluation_instances_all = []
    for seq in evaluation_sequence:
        sequence_name = sequence_dict[seq]
        label_gt_with_dynamic_sub_folder = os.path.join(gt_label_path_with_dynamic,sequence_name,"image_00/data_rect/")
        assert os.path.exists(label_gt_with_dynamic_sub_folder)
        for label_name in sorted(os.listdir(label_gt_with_dynamic_sub_folder)):
            label_name_completed = os.path.join(label_gt_with_dynamic_sub_folder,label_name)
            evaluation_instances_all.append(label_name_completed)
            
    # ------------------------------------Training and the Validation Set------------------------------------------------------------
    # Begin soft-link here for the training and the valdaition set.
    IS_TRAINING = False
    idx = 0
    sync_training_list = []
    for fname in tqdm(trainval_instances_all):
        gt_label_path = fname
        est_label_path = gt_label_path.replace("my_gts_with_dynamic","predictions")
        assert est_label_path!=gt_label_path
        basename_with_sequence = extract_from_2013_regex(gt_label_path)[:-4]
        source_image_2 = os.path.join(root_dirname,"data_2d_raw",basename_with_sequence)+".png"
        source_image_3 = source_image_2.replace("image_00","image_01")
        camera_pose = os.path.join(root_dirname,"cam_calib.txt")
        
        assert os.path.exists(source_image_2)
        assert os.path.exists(source_image_3)
        assert os.path.exists(gt_label_path)
        assert os.path.exists(est_label_path)
        assert os.path.exists(camera_pose)
        
        
        if fname in training_instances_all:
            IS_TRAINING = True
        elif fname in validation_instances_all:
            IS_TRAINING = False
            
            
        base_filename = f"{str(idx).zfill(6)}"
            
        training_sync_target_image_2 = os.path.join(output_folder_training_image_2_folder_path,base_filename+".png")
        training_sync_target_image_3 = os.path.join(output_folder_training_image_3_folder_path,base_filename+".png")
        training_sync_target_label_2 = os.path.join(output_folder_training_label_2_folder_path,base_filename+".txt")
        training_sync_target_label_gt = os.path.join(output_folder_training_label_gt_folder_path,base_filename+".txt")
        training_sync_target_calib_label = os.path.join(output_folder_training_calib_folder_path,base_filename+".txt")
        

        if IS_TRAINING:
            training_filename_splits_ImageSets_list.append(base_filename)
        else:
            validation_filename_splits_ImageSets_list.append(base_filename)
        trainval_filename_splits_ImageSets_list.append(base_filename)


        # sync_link image 2
        if not os.path.exists(training_sync_target_image_2):
            os.system("ln -s {} {}".format(source_image_2,training_sync_target_image_2))
        # sync_link image 3
        if not os.path.exists(training_sync_target_image_3):
            os.system("ln -s {} {}".format(source_image_3,training_sync_target_image_3))       
        # label 2
        if not os.path.exists(training_sync_target_label_2):
            shutil.copy(est_label_path,training_sync_target_label_2)
        # label gt
        if not os.path.exists(training_sync_target_label_gt):
            shutil.copy(gt_label_path,training_sync_target_label_gt)
        
        # calib
        if not os.path.exists(training_sync_target_calib_label):
            shutil.copy(camera_pose,training_sync_target_calib_label)


        # create the sync list
        current_line = source_image_2 + " " + training_sync_target_image_2 + " " + source_image_3 + " " + training_sync_target_image_3 + " " + gt_label_path + " " + training_sync_target_label_gt
        sync_training_list.append(current_line)
        
        idx = idx + 1
        
    saved_training_sync_name = os.path.join(output_folder_training,'sync_file.txt')
    with open(saved_training_sync_name,'w') as f:
        for idx, line in enumerate(sync_training_list):
            if idx!=len(sync_training_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)

    saved_into_txt(saved_name=training_splits_filename_path_sync,
                   content_list=training_filename_splits_ImageSets_list)
    
    saved_into_txt(saved_name=validation_splits_filename_path_sync,
                   content_list=validation_filename_splits_ImageSets_list)
    
    saved_into_txt(saved_name=training_validation_splits_filename_path_sync,
                   content_list=trainval_filename_splits_ImageSets_list)

    # --------------------------------------Testing Set ------------------------------------------------------#
    idx = 0
    sync_testing_list = []
    for fname in tqdm(evaluation_instances_all):
        gt_label_path = fname
        est_label_path = gt_label_path.replace("my_gts_with_dynamic","predictions")
        assert est_label_path!=gt_label_path
        basename_with_sequence = extract_from_2013_regex(gt_label_path)[:-4]
        source_image_2 = os.path.join(root_dirname,"data_2d_raw",basename_with_sequence)+".png"
        source_image_3 = source_image_2.replace("image_00","image_01")
        camera_pose = os.path.join(root_dirname,"cam_calib.txt")
        
        assert os.path.exists(source_image_2)
        assert os.path.exists(source_image_3)
        assert os.path.exists(gt_label_path)
        assert os.path.exists(est_label_path)
        assert os.path.exists(camera_pose)


        base_filename = f"{str(idx).zfill(6)}"
            
        testing_sync_target_image_2 = os.path.join(output_folder_testing_image_2_folder_path,base_filename+".png")
        testing_sync_target_image_3 = os.path.join(output_folder_testing_image_3_folder_path,base_filename+".png")
        testing_sync_target_label_2 = os.path.join(output_folder_testing_label_2_folder_path,base_filename+".txt")
        testing_sync_target_label_gt = os.path.join(output_folder_testing_label_gt_folder_path,base_filename+".txt")
        testing_sync_target_calib_label = os.path.join(output_folder_testing_calib_folder_path,base_filename+".txt")
        


        # sync_link image 2
        if not os.path.exists(testing_sync_target_image_2):
            os.system("ln -s {} {}".format(source_image_2,testing_sync_target_image_2))
        # sync_link image 3
        if not os.path.exists(testing_sync_target_image_3):
            os.system("ln -s {} {}".format(source_image_3,testing_sync_target_image_3))       
        # label 2
        if not os.path.exists(testing_sync_target_label_2):
            shutil.copy(est_label_path,testing_sync_target_label_2)
        # label gt
        if not os.path.exists(testing_sync_target_label_gt):
            shutil.copy(gt_label_path,testing_sync_target_label_gt)
        
        # calib
        if not os.path.exists(testing_sync_target_calib_label):
            shutil.copy(camera_pose,testing_sync_target_calib_label)


        # create the sync list
        current_line = source_image_2 + " " + testing_sync_target_image_2 + " " + source_image_3 + " " + testing_sync_target_image_3 + " " + gt_label_path + " " + testing_sync_target_label_gt
        sync_testing_list.append(current_line)
        
        idx = idx + 1
        
        
    saved_testing_sync_name = os.path.join(output_folder_testing,'sync_file.txt')
    with open(saved_testing_sync_name,'w') as f:
        for idx, line in enumerate(sync_testing_list):
            if idx!=len(sync_testing_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)


    saved_into_txt(saved_name=testing_splits_filename_path_sync,
                   content_list=testing_filename_splits_ImageSets_list)
        
        
        
        
        
        

        

    
    
    
    # Begin soft-link here for the testing set.
    
    

    