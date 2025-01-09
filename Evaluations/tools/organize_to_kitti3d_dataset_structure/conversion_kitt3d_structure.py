import os
import argparse
import shutil
from tqdm import tqdm


def main(args):
    
    root_dirname = args.root_dirname
    prediction_label_path = args.prediction_label_path
    gt_label_path = args.gt_label_path
    training_split = (args.training_split).split(",")
    testing_split = (args.testing_split).split(",")
    
    output_folder = args.output_folder
    output_folder_training = os.path.join(output_folder,'training')
    output_folder_testing = os.path.join(output_folder,'testing')
    
    output_folder_training_image_2_folder_path = os.path.join(output_folder_training,'image_2')
    output_folder_training_image_3_folder_path = os.path.join(output_folder_training,'image_3')
    output_folder_training_calib_folder_path = os.path.join(output_folder_training,"calib")
    output_folder_training_label_2_folder_path = os.path.join(output_folder_training,'label_2')
    output_folder_training_label_est_folder_path = os.path.join(output_folder_training,'label_est')
    

    output_folder_testing_image_2_folder_path = os.path.join(output_folder_testing,'image_2')
    output_folder_testing_image_3_folder_path = os.path.join(output_folder_testing,'image_3')
    output_folder_testing_calib_folder_path = os.path.join(output_folder_testing,"calib")
    output_folder_testing_label_2_folder_path = os.path.join(output_folder_testing,'label_2')
    output_folder_testing_label_est_folder_path = os.path.join(output_folder_testing,'label_est')
    
    
    os.makedirs(output_folder_training_image_2_folder_path,exist_ok=True)
    os.makedirs(output_folder_training_image_3_folder_path,exist_ok=True)
    os.makedirs(output_folder_training_calib_folder_path,exist_ok=True)
    os.makedirs(output_folder_training_label_2_folder_path,exist_ok=True)
    os.makedirs(output_folder_training_label_est_folder_path,exist_ok=True)
    
    
    os.makedirs(output_folder_testing_image_2_folder_path,exist_ok=True)
    os.makedirs(output_folder_testing_image_3_folder_path,exist_ok=True)
    os.makedirs(output_folder_testing_calib_folder_path,exist_ok=True)
    os.makedirs(output_folder_testing_label_2_folder_path,exist_ok=True)
    os.makedirs(output_folder_testing_label_est_folder_path,exist_ok=True)
    
    
    
    
    sequence_list = sorted(['2013_05_28_drive_0003_sync', 
                     '2013_05_28_drive_0009_sync', 
                     '2013_05_28_drive_0005_sync', 
                     '2013_05_28_drive_0004_sync', 
                     '2013_05_28_drive_0006_sync', 
                     '2013_05_28_drive_0002_sync', 
                     '2013_05_28_drive_0010_sync', 
                     '2013_05_28_drive_0000_sync', 
                     '2013_05_28_drive_0007_sync'])
    
    training_splits_list = []
    testing_splits_list = []
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
        

    # Move to training file structure
    idx = 0
    sync_training_list = []
    for seq in training_splits_list:
        label_2_folder = os.path.join(gt_label_path,seq,"image_00/data_rect/")
        for label_name in tqdm(sorted(os.listdir(label_2_folder))):
            
            base_filename = f"{str(idx).zfill(6)}"

            # copy label_2
            label_2_name_abs = os.path.join(label_2_folder,label_name)

            assert os.path.exists(label_2_name_abs)
            saved_label_2 = os.path.join(output_folder_training_label_2_folder_path,base_filename+".txt")
            if not os.path.exists(saved_label_2):
                shutil.copy(label_2_name_abs, saved_label_2)
            
            # copy label_est
            label_est_name_abs = label_2_name_abs.replace(gt_label_path,prediction_label_path)
            
            if not os.path.exists(label_est_name_abs):
                continue
            assert os.path.exists(label_est_name_abs)
            saved_label_est = os.path.join(output_folder_training_label_est_folder_path,base_filename+".txt")
            if not os.path.exists(saved_label_est):
                shutil.copy(label_est_name_abs, saved_label_est)
            
            
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
            current_line = source_image_filename_left + " " + saved_image_left + " " + label_est_name_abs + " " + saved_label_est + " " + label_2_name_abs + " " + saved_label_2
            sync_training_list.append(current_line)
            idx = idx + 1
        
    
    saved_training_sync_name = os.path.join(output_folder_training,'sync_file.txt')
    with open(saved_training_sync_name,'w') as f:
        for idx, line in enumerate(sync_training_list):
            if idx!=len(sync_training_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)


    # Move to testing file structure
    idx = 0
    sync_testing_list = []
    for seq in testing_splits_list:
        label_2_folder = os.path.join(gt_label_path,seq,"image_00/data_rect/")
        for label_name in tqdm(sorted(os.listdir(label_2_folder))):
            
            base_filename = f"{str(idx).zfill(6)}"

            # copy label_2
            label_2_name_abs = os.path.join(label_2_folder,label_name)
            assert os.path.exists(label_2_name_abs)
            saved_label_2 = os.path.join(output_folder_testing_label_2_folder_path,base_filename+".txt")
            if not os.path.exists(saved_label_2):
                shutil.copy(label_2_name_abs, saved_label_2)
            
            # copy label_est
            label_est_name_abs = label_2_name_abs.replace(gt_label_path,prediction_label_path)
            if not os.path.exists(label_est_name_abs):
                continue
            assert os.path.exists(label_est_name_abs)
            saved_label_est = os.path.join(output_folder_testing_label_est_folder_path,base_filename+".txt")
            if not os.path.exists(saved_label_est):
                shutil.copy(label_est_name_abs, saved_label_est)
            
            
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
            current_line = source_image_filename_left + " " + saved_image_left + " " + label_est_name_abs + " " + saved_label_est + " " + label_2_name_abs + " " + saved_label_2

            
            sync_testing_list.append(current_line)
            idx = idx + 1
        
    
    saved_testing_sync_name = os.path.join(output_folder_testing,'sync_file.txt')
    with open(saved_testing_sync_name,'w') as f:
        for idx, line in enumerate(sync_testing_list):
            if idx!=len(sync_testing_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)






if __name__=="__main__":


    parser = argparse.ArgumentParser(description="Convert VSRD Format into KITTI3D Dataset Configuration")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-360")
    parser.add_argument("--prediction_label_path",type=str, default="datasets/KITTI-360")
    parser.add_argument("--gt_label_path",type=str,default="datasets/KITTI-360")
    parser.add_argument("--training_split",type=str,default="dataset")
    parser.add_argument("--testing_split",type=str,default='daatset')
    parser.add_argument("--output_folder",type=str,default='daatset')


    args = parser.parse_args()
    
    main(args=args)