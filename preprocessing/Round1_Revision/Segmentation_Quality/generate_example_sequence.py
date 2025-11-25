import random
import os 

def read_txt_files_lines_as_list(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()
    
def save_list_to_txt(list, file_path):
    with open(file_path, 'w') as file:
        for item in list:
            file.write(item + "\n")



if __name__=="__main__":
    root_folder = "/data/dataset/KITTI/KITTI360_For_Docker"    
    sample_txt_filename  = "/data/dataset/VSRD_PP_Sync/train_ablation_filenames/train_ablation_filenames.txt"
    dynamic_txt_filename = "/data/dataset/VSRD_PP_Sync/train_ablation_filenames/train_ablation_dynamic_mask.txt"

    readed_sample_txt_lines = read_txt_files_lines_as_list(sample_txt_filename)
    readed_dynamic_txt_lines = read_txt_files_lines_as_list(dynamic_txt_filename)
    
    
    # random sample 10 lines (same) from the readed_sample_txt_lines and readed_dynamic_txt_lines
    random_sample_sample_txt_lines = random.sample(readed_sample_txt_lines, 10)
    sample_fname_list = []
    dynamic_fname_list = []
    
    
    for idx, line in enumerate(random_sample_sample_txt_lines):
        
        splits = line.split(" ")
        instance_ids = splits[0]
        image_fname = splits[1]
        relative_index = splits[2]
        assert os.path.exists(image_fname)
        sample_fname_list.append(line)
        
        for dynamic_line in readed_dynamic_txt_lines:
            dynamic_instance_ids, dynamic_image_fname, dynamic_labels = dynamic_line.split(" ")        
            if dynamic_image_fname == image_fname:
                dynamic_fname_list.append(dynamic_line)
                break
    
    saved_folder_sample_folder = "/data/dataset/VSRD_PP_Sync/Round1_Revision/Selected_Abaltions/train_ablation_filenames_example.txt"
    saved_folder_dynamic_folder = "/data/dataset/VSRD_PP_Sync/Round1_Revision/Selected_Abaltions/train_ablation_dynamic_mask_example.txt"
    os.makedirs(os.path.dirname(saved_folder_sample_folder),exist_ok=True)
    os.makedirs(os.path.dirname(saved_folder_dynamic_folder),exist_ok=True)


    save_list_to_txt(sample_fname_list, 
                     saved_folder_sample_folder)
    
    save_list_to_txt(dynamic_fname_list, 
                     saved_folder_dynamic_folder)
        
    

