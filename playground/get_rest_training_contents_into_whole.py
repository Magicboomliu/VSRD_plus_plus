import os
import numpy as np

from tqdm import tqdm

# read the text contents from the `.txt`
def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def start_with_data_2d_raw(string):
    return string[string.index("data_2d_raw"):]


def write_into_txt(filename,lst):
    with open(filename,'w') as f:
        for idx, line in enumerate(lst):
            if idx!=len(lst)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)


if __name__=="__main__":
    
    # All templated Path
    templated_sample_name_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/filenames/R50-N16-M128-B16/2013_05_28_drive_0000_sync/sampled_image_filenames.txt"
    templated_dynamic_name_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/est_dynamic_list/sync00/dynamic_mask.txt"
    assert os.path.exists(templated_sample_name_path)
    assert os.path.exists(templated_dynamic_name_path)
    estimated_velocities_folder = "/media/zliu/data12/TPAMI_Results/EST_VELO/Estimated_Velocites/"
    assert os.path.exists(estimated_velocities_folder)
    
    
    await_processed_sequnces_names_list = ["00","02","03","04","05","06","07","09","10"]
    
    
    missed_sample_name_path_list = []
    missed_dynamic_name_path_list = []
    
    
    for seq_name in tqdm(await_processed_sequnces_names_list):
        current_sample_name_path = templated_sample_name_path.replace("2013_05_28_drive_0000_sync","2013_05_28_drive_00{}_sync".format(seq_name))
        assert os.path.exists(current_sample_name_path)
        current_dynamic_name_path = templated_dynamic_name_path.replace("sync00","sync{}".format(seq_name))
        assert os.path.exists(current_dynamic_name_path)
        
        current_sample_contents = read_text_lines(current_sample_name_path)
        current_dynamic_contents = read_text_lines(current_dynamic_name_path)

        assert len(current_sample_contents) == len(current_dynamic_contents)
        
        for idx, line in enumerate(current_sample_contents):
            splits = line.split()
            current_fname = splits[1]
            saved_velocity_fname = os.path.join(estimated_velocities_folder,start_with_data_2d_raw(current_fname).replace(".png",".pkl"))
            if os.path.exists(saved_velocity_fname):
                pass
            else:
                missed_sample_name_path_list.append(line)
                missed_dynamic_name_path_list.append(current_dynamic_contents[idx])
        
    
    
    write_into_txt(filename="missed_sample_filenameV2.txt",lst=missed_sample_name_path_list)
    write_into_txt(filename="missed_dynamic_filenameV2.txt",lst=missed_dynamic_name_path_list)
    

                


    