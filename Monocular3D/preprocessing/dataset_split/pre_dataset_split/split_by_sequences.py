import os
import numpy as np
from tqdm import tqdm
import argparse
import shutil



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
    parser.add_argument("--pseudo_labels_dirname", type=str, default="")
    parser.add_argument("--training_split",type=str,default="dataset")
    parser.add_argument("--valiation_split",type=str,default="dataset")
    parser.add_argument("--testing_split",type=str,default='daatset')
    parser.add_argument("--output_dataset_path",type=str,default='daatset')
    parser.add_argument("--output_dataset_splits_filename_path",type=str,default='')
    args = parser.parse_args()
    

    root_dirname = args.root_dirname
    
    prediction_label_path = os.path.join(args.pseudo_labels_dirname,"predictions")
    gt_label_path_with_dynamic = os.path.join(args.pseudo_labels_dirname,"my_gts_with_dynamic")
    training_split = (args.training_split).split(",")
    validation_split = (args.valiation_split).split(",")
    testing_split = (args.testing_split).split(",")
    
    
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
    
    
    
    
    output_folder = args.output_dataset_path
    output_folder_training = os.path.join(output_folder,'training')
    output_folder_testing = os.path.join(output_folder,'testing')
    train_val_split = training_split + validation_split
    train_val_split = sorted(train_val_split)


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


    sequence_list = sorted(['2013_05_28_drive_0003_sync', 
                     '2013_05_28_drive_0009_sync', 
                     '2013_05_28_drive_0005_sync', 
                     '2013_05_28_drive_0004_sync', 
                     '2013_05_28_drive_0006_sync', 
                     '2013_05_28_drive_0002_sync', 
                     '2013_05_28_drive_0010_sync', 
                     '2013_05_28_drive_0000_sync', 
                     '2013_05_28_drive_0007_sync'])
    
    
    
    # training / validation splits
    training_validation_splits_list = []
    for seq in train_val_split:
        if seq =='00':
            training_validation_splits_list.append('2013_05_28_drive_0000_sync')
        elif seq=='02':
            training_validation_splits_list.append('2013_05_28_drive_0002_sync')
        elif seq =='03':
            training_validation_splits_list.append('2013_05_28_drive_0003_sync')
        elif seq =='04':
            training_validation_splits_list.append('2013_05_28_drive_0004_sync')
        elif seq =='05':
            training_validation_splits_list.append('2013_05_28_drive_0005_sync')
        elif seq =='06':
            training_validation_splits_list.append('2013_05_28_drive_0006_sync')
        elif seq =='07':
            training_validation_splits_list.append('2013_05_28_drive_0007_sync')
        elif seq =='09':
            training_validation_splits_list.append('2013_05_28_drive_0009_sync')
        elif seq =='10':
            training_validation_splits_list.append('2013_05_28_drive_0010_sync')

    # training splits
    training_splits_list = []
    for seq in training_split:
        if seq =='00':
            training_splits_list.append('2013_05_28_drive_0000_sync')
        elif seq=='02':
            training_splits_list.append('2013_05_28_drive_0002_sync')
        elif seq =='03':
            training_splits_list.append('2013_05_28_drive_0003_sync')
        elif seq =='04':
            training_splits_list.append('2013_05_28_drive_0004_sync')
        elif seq =='05':
            training_splits_list.append('2013_05_28_drive_0005_sync')
        elif seq =='06':
            training_splits_list.append('2013_05_28_drive_0006_sync')
        elif seq =='07':
            training_splits_list.append('2013_05_28_drive_0007_sync')
        elif seq =='09':
            training_splits_list.append('2013_05_28_drive_0009_sync')
        elif seq =='10':
            training_splits_list.append('2013_05_28_drive_0010_sync')
    
    # validation splits
    validation_splits_list = []
    for sequence_name in training_validation_splits_list:
        if sequence_name in training_splits_list:
            pass
        else:
            validation_splits_list.append(sequence_name)


    # testing splits
    testing_splits_list = []
    for seq in testing_split:
        if seq =='00':
            testing_splits_list.append('2013_05_28_drive_0000_sync')
        elif seq=='02':
            testing_splits_list.append('2013_05_28_drive_0002_sync')
        elif seq =='03':
            testing_splits_list.append('2013_05_28_drive_0003_sync')
        elif seq =='04':
            testing_splits_list.append('2013_05_28_drive_0004_sync')
        elif seq =='05':
            testing_splits_list.append('2013_05_28_drive_0005_sync')
        elif seq =='06':
            testing_splits_list.append('2013_05_28_drive_0006_sync')
        elif seq =='07':
            testing_splits_list.append('2013_05_28_drive_0007_sync')
        elif seq =='09':
            testing_splits_list.append('2013_05_28_drive_0009_sync')
        elif seq =='10':
            testing_splits_list.append('2013_05_28_drive_0010_sync')
        

    # Create Soft-Link For the Training and Validation Splits
    IS_TRAINING = False
    idx = 0
    sync_training_list = []
    for seq in tqdm(training_validation_splits_list):
        # in the training splits
        if seq in training_splits_list:
            IS_TRAINING = True
        # in the validation  splits
        elif seq in validation_splits_list:
            IS_TRAINING = False
        else:
            raise FileNotFoundError
        
        label_2_folder = os.path.join(prediction_label_path,seq,"image_00/data_rect/") # label_2
        assert os.path.exists(label_2_folder)
        for label_name in sorted(os.listdir(label_2_folder)):
            base_filename = f"{str(idx).zfill(6)}"
            
            
            if IS_TRAINING:
                training_filename_splits_ImageSets_list.append(base_filename)
            else:
                validation_filename_splits_ImageSets_list.append(base_filename)
            trainval_filename_splits_ImageSets_list.append(base_filename)
            

            # copy label_2
            label_2_name_abs = os.path.join(label_2_folder,label_name)
            assert os.path.exists(label_2_name_abs)
            saved_label_2 = os.path.join(output_folder_training_label_2_folder_path,base_filename+".txt")
            if not os.path.exists(saved_label_2):
                shutil.copy(label_2_name_abs, saved_label_2)
            
            # copy label_gt
            label_gt_name_abs = label_2_name_abs.replace(prediction_label_path,gt_label_path_with_dynamic)
            assert os.path.exists(label_gt_name_abs)
            
            saved_label_gt = os.path.join(output_folder_training_label_gt_folder_path,base_filename+".txt")
            if not os.path.exists(saved_label_gt):
                shutil.copy(label_gt_name_abs, saved_label_gt)
            
            
            # copy the calib
            source_calib_filename = os.path.join(root_dirname,"cam_calib.txt")
            assert os.path.exists(source_calib_filename)
            saved_calib_est = os.path.join(output_folder_training_calib_folder_path,base_filename+".txt")
            if not os.path.exists(saved_calib_est):
                shutil.copy(source_calib_filename,saved_calib_est)
                
            
            # soft link the images    
            source_image_filename_left = os.path.join(root_dirname,"data_2d_raw",seq,"image_00/data_rect/",label_name.replace(".txt",".png"))
            
            assert os.path.exists(source_image_filename_left)
            saved_image_left = os.path.join(output_folder_training_image_2_folder_path,base_filename+".png")
            if not os.path.exists(saved_image_left):
                os.system("ln -s {} {}".format(source_image_filename_left,saved_image_left))
                
            
            source_image_filename_right = source_image_filename_left.replace("image_00","image_01")
            assert os.path.exists(source_image_filename_right)
            saved_image_right = os.path.join(output_folder_training_image_3_folder_path,base_filename+".png")
            if not os.path.exists(saved_image_right):
                os.system("ln -s {} {}".format(source_image_filename_right,saved_image_right))
                
                
            # create the sync list
            current_line = source_image_filename_left + " " + saved_image_left + " " + label_gt_name_abs + " " + saved_label_gt + " " + label_2_name_abs + " " + saved_label_2
            sync_training_list.append(current_line)
            idx = idx + 1
        
    saved_training_sync_name = os.path.join(output_folder_training,'sync_file.txt')
    with open(saved_training_sync_name,'w') as f:
        for idx, line in enumerate(sync_training_list):
            if idx!=len(sync_training_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)

    # Create Soft-Link for the Testing Splits
    # Move to testing file structure
    idx = 0
    sync_testing_list = []
    for seq in tqdm(testing_splits_list):
        label_2_folder = os.path.join(prediction_label_path,seq,"image_00/data_rect/")
        for label_name in sorted(os.listdir(label_2_folder)):
            
            base_filename = f"{str(idx).zfill(6)}"

            testing_filename_splits_ImageSets_list.append(base_filename)

            # copy label_2
            label_2_name_abs = os.path.join(label_2_folder,label_name)
            assert os.path.exists(label_2_name_abs)
            saved_label_2 = os.path.join(output_folder_testing_label_2_folder_path,base_filename+".txt")
            if not os.path.exists(saved_label_2):
                shutil.copy(label_2_name_abs, saved_label_2)
            
            # copy label_gt
            label_gt_name_abs = label_2_name_abs.replace(prediction_label_path,gt_label_path_with_dynamic)
            assert os.path.exists(label_gt_name_abs)
            saved_label_gt = os.path.join(output_folder_testing_label_gt_folder_path,base_filename+".txt")
            if not os.path.exists(saved_label_gt):
                shutil.copy(label_gt_name_abs, saved_label_gt)
            
            
            # copy the calib
            source_calib_filename = os.path.join(root_dirname,"cam_calib.txt")
            assert os.path.exists(source_calib_filename)
            saved_calib_est = os.path.join(output_folder_testing_calib_folder_path,base_filename+".txt")
            if not os.path.exists(saved_calib_est):
                shutil.copy(source_calib_filename,saved_calib_est)
                
            
            # soft link the images    
            source_image_filename_left = os.path.join(root_dirname,"data_2d_raw",seq,"image_00/data_rect/",label_name.replace(".txt",".png"))
            assert os.path.exists(source_image_filename_left)
            saved_image_left = os.path.join(output_folder_testing_image_2_folder_path,base_filename+".png")
            if not os.path.exists(saved_image_left):
                os.system("ln -s {} {}".format(source_image_filename_left,saved_image_left))
                
            
            source_image_filename_right = source_image_filename_left.replace("image_00","image_01")
            assert os.path.exists(source_image_filename_right)
            saved_image_right = os.path.join(output_folder_testing_image_3_folder_path,base_filename+".png")
            if not os.path.exists(saved_image_right):
                os.system("ln -s {} {}".format(source_image_filename_right,saved_image_right))
                
            # create the sync list
            current_line = source_image_filename_left + " " + saved_image_left + " " + label_gt_name_abs + " " + saved_label_gt + " " + label_2_name_abs + " " + saved_label_2
            sync_testing_list.append(current_line)
            idx = idx + 1
        
    
    saved_testing_sync_name = os.path.join(output_folder_testing,'sync_file.txt')
    with open(saved_testing_sync_name,'w') as f:
        for idx, line in enumerate(sync_testing_list):
            if idx!=len(sync_testing_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)


    saved_into_txt(saved_name=training_splits_filename_path_sync,
                   content_list=training_filename_splits_ImageSets_list)
    
    saved_into_txt(saved_name=validation_splits_filename_path_sync,
                   content_list=validation_filename_splits_ImageSets_list)
    
    saved_into_txt(saved_name=training_validation_splits_filename_path_sync,
                   content_list=trainval_filename_splits_ImageSets_list)
    
    saved_into_txt(saved_name=testing_splits_filename_path_sync,
                   content_list=testing_filename_splits_ImageSets_list)
    
    
    







    
    

    # sync_filepath_trainval = os.path.join(args.root_dirname,'training','sync_file.txt')
    # sync_filepath_testing = os.path.join(args.root_dirname,'testing','sync_file.txt')
    # train_list_splits = args.training_split
    # val_list_splits = args.valiation_split
    # test_list_splits = args.testing_split
    # saved_location = args.output_dataset_path
    # os.makedirs(saved_location,exist_ok=True)
    
    # sequence_dict = {"00":"2013_05_28_drive_0000_sync",
    #                  '02':"2013_05_28_drive_0002_sync",
    #                  '03':"2013_05_28_drive_0003_sync",
    #                  '04':"2013_05_28_drive_0004_sync",
    #                  '06':"2013_05_28_drive_0006_sync",
    #                  '05':"2013_05_28_drive_0005_sync",
    #                  '07': "2013_05_28_drive_0007_sync",
    #                  '09':"2013_05_28_drive_0009_sync",
    #                  '10':"2013_05_28_drive_0010_sync"
    #                  }
    
    # train_seq_list = []
    # val_seq_list = []
    # trainval_seq_list = []
    # test_seq_list = []
    
    
    # for seq in train_list_splits.split(","):
    #     train_seq_list.append(sequence_dict[seq])
        
    # for seq in val_list_splits.split(","):
    #     val_seq_list.append(sequence_dict[seq])
    #     trainval_seq_list = train_seq_list + val_seq_list
    
    # for seq in test_list_splits.split(","):
    #     test_seq_list.append(sequence_dict[seq])
      
    
    # train_contents = []
    # val_contents = []
    # train_val_contents = []
    
    # # training contents and val contents creation
    # contents = read_text_lines(sync_filepath_trainval)
    # for line in tqdm(contents):
    #     source_image_filename_left,saved_image_left, label_est_name_abs,saved_label_est,label_2_name_abs, saved_label_2= line.split(" ")
    #     current_seq = starts_with_2013(source_image_filename_left.split("/"))
    #     basename = os.path.basename(saved_image_left)[:-4]
        
    #     if current_seq in train_seq_list:
    #         train_contents.append(basename)
    #     elif current_seq in val_seq_list:
    #         val_contents.append(basename)
    #     elif current_seq in test_seq_list:
    #         raise NotImplementedError
    #     else:
    #         raise NotImplementedError
        
    # train_val_contents = train_contents + val_contents
    # # testing contents creation
    # test_contents = []
    # # training contents and val contents creation
    # contents = read_text_lines(sync_filepath_testing)
    # for line in tqdm(contents):
    #     source_image_filename_left,saved_image_left, label_est_name_abs,saved_label_est,label_2_name_abs, saved_label_2= line.split(" ")
    #     current_seq = starts_with_2013(source_image_filename_left.split("/"))
    #     basename = os.path.basename(saved_image_left)[:-4]
    #     if current_seq in test_seq_list:
    #         test_contents.append(basename)
    #     else:
    #         raise NotImplementedError
        
        
    
    # saved_training_path = os.path.join(saved_location,'train.txt')
    # saved_valiation_path = os.path.join(saved_location,'val.txt')
    # saved_trainval_path = os.path.join(saved_location,'trainval.txt')
    # saved_test_path = os.path.join(saved_location,'test.txt')
    
    # saved_into_txt(saved_name=saved_training_path,
    #                content_list=train_contents)
    
    # saved_into_txt(saved_name=saved_valiation_path,
    #                content_list=val_contents)
    # saved_into_txt(saved_name=saved_trainval_path,
    #                content_list=train_val_contents)
    # saved_into_txt(saved_name=saved_test_path,
    #                content_list=test_contents)
    
    

    
    
    
        

    