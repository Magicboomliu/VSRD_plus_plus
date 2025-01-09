import os
import sys
import shutil

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines



if __name__=="__main__":
    
    input_folder = "/media/zliu/data12/dataset/TPAMI_Ablations/Synced_For_Evaluations/VSRD_Vanilla/training/label_2/"
    output_folder = "/media/zliu/data12/dataset/TPAMI_Ablations/Synced_For_Evaluations/VSRD_Vanilla/training/label_2_for_vis/"
    os.makedirs(output_folder,exist_ok=True)
    sync_file = input_folder.replace("label_2/","sync_file.txt")
    
    contents = read_text_lines(sync_file)
    for line in contents:
        splits = line.split()
        gt_image = splits[0]
        sync_image = splits[1]
        
        sync_label_original = sync_image.replace("image_2","label_2").replace(".png",".txt")
        saved_label_path = sync_label_original.replace("label_2","label_2_for_vis")
        
        if "drive_0003" in gt_image or "drive_0007" in gt_image:
            
            shutil.copy(src=sync_label_original,dst=saved_label_path)
        

    
    
    pass