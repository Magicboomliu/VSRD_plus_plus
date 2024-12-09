import os
import numpy as np
from tqdm import tqdm

# read contents
def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines



# update the filenames list
# update the dynamic list

if __name__=="__main__":

    training_filenames_path = "/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/train_tsubame_filenames/train_all_filenames.txt"
    dynamic_filenames_path = "/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/train_tsubame_filenames/train_all_dyanmic_filenames.txt"
    current_folder_name_path = "/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/output_models/ckpts"

    

    training_filenames_path_new = "/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/train_tsubame_filenames/train_all_filenames_V1.txt"
    dynamic_filenames_path_new = "/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/train_tsubame_filenames/train_all_dyanmic_filenames_V1.txt"


    
    
    assert os.path.exists(training_filenames_path)
    assert os.path.exists(dynamic_filenames_path)
    
    training_contents = read_text_lines(training_filenames_path)
    dynamic_contents = read_text_lines(dynamic_filenames_path)
    assert len(training_contents) == len(dynamic_contents)
    
    
    lefted_training_filename_list = []
    lefted_training_dynamic_list = []
    
    finished_trainig_filename_list = []
    finished_training_dynamic_list = []
    
    for idx, content in enumerate(training_contents):

        sample_filename = content.split(" ")[1]
        dyanmic_filename = dynamic_contents[idx].split(" ")[1]

        
        start_index = sample_filename.find("data_2d_raw")
        filename_dirname = sample_filename[start_index:][:-4]
        
        current_ckpt_folder = os.path.join(current_folder_name_path,filename_dirname)
        ckpt_file_path= os.path.join(current_ckpt_folder,'step_2499.pt')

        assert sample_filename == dyanmic_filename
    
        
        if os.path.exists(ckpt_file_path):
            finished_trainig_filename_list.append(content)
            finished_training_dynamic_list.append(dynamic_contents[idx])
        else:
            
            lefted_training_filename_list.append(content)
            lefted_training_dynamic_list.append(dynamic_contents[idx])
    
    
    print(len(finished_trainig_filename_list)/len(training_contents) * 100)
    print(len(finished_training_dynamic_list)/len(dynamic_contents) * 100)

    with open(training_filenames_path_new,'w') as f:
        for idx, line in enumerate(lefted_training_filename_list):
            if idx!=len(lefted_training_filename_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)

    with open(dynamic_filenames_path_new,'w') as f:
        for idx, line in enumerate(lefted_training_dynamic_list):
            if idx!=len(lefted_training_dynamic_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)
            

