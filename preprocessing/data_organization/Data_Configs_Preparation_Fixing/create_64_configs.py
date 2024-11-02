import shutil
import os
import sys

if __name__=="__main__":
    
    configs_root_path = "/home/zliu/CVPR25_Detection/Submitted_Version/VSRD-V2/Optimized_Based/configs/SPLITS64"
    original_file = "train_config_sub_00.py"
    original_file = os.path.join(configs_root_path,original_file)
    
    N_SPLITS = 64
    
    for i in range(N_SPLITS):
        if i ==0:
            continue
        suffix = f"{i:02}"
        new_file = f"train_config_sub_{suffix}.py"
        new_file = os.path.join(configs_root_path,new_file)
    
        shutil.copy(original_file, new_file)
    
    print("Done")

