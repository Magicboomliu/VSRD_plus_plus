import os
import sys
from tqdm import tqdm
import re

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def find_indices(A, B):
    indices = []
    for b in B:
        indices.append(A.index(b))
    return indices



def process_the_data(dynamic_mask_list_path,sample_group_list_path,output_path):
    
    dynamic_list_contents = read_text_lines(dynamic_mask_list_path)
    sample_group_contents = read_text_lines(sample_group_list_path)
    saved_contents = []
    for idx in range(len(dynamic_list_contents)):
        # dynamic contents
        dynamic_content = dynamic_list_contents[idx]
        instance_ids_dynamic, image_fname_dynamic, dynamic_labels = dynamic_content.strip().split(" ")
        
        # sampled contents
        sample_content = sample_group_contents[idx]
        instance_ids_sample, image_fname_sample,adja_frames = sample_content.strip().split(" ")
        
        assert image_fname_dynamic == image_fname_sample
        assert len(instance_ids_dynamic)>=len(instance_ids_sample)
        
        # matching and get the masks
        
        instance_ids_dynamic_list = instance_ids_dynamic.split(",")
        instance_ids_sample_list = instance_ids_sample.split(",")
        dynamic_labels_list = dynamic_labels.split(",")
        

        indices = find_indices(instance_ids_dynamic_list, instance_ids_sample_list)
        
        instance_ids_dynamic_list = [instance_ids_dynamic_list[i] for i in indices]
        dynamic_labels_list = [dynamic_labels_list[i] for i in indices]
        
        assert len(instance_ids_dynamic_list) == len(dynamic_labels_list)
        

        
        instance_ids_dynamic =  ','.join(instance_ids_dynamic_list)
        dynamic_labels = ",".join(dynamic_labels_list)
        
            
        assert len(instance_ids_dynamic)==len(instance_ids_sample)

        saved_contents.append(instance_ids_dynamic+" "+image_fname_dynamic+" "+dynamic_labels)

    
    with open(output_path,'w') as f:
        for idx, line in enumerate(saved_contents):
            if idx!=len(saved_contents)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)

    
    
if __name__=="__main__":
    
    input_list =['00','02','03','04','05','06','07','09','10']

    for name in input_list:
        dynamic_mask_list_path = "/data3/VSRD_PP_Sync/estimated_dynamic_static_filenames/sync{}/dynamic_mask.txt".format(name)
        sample_group_list_path = "/data3/VSRD_PP_Sync/filenames/R50-N16-M128-B16/2013_05_28_drive_00{}_sync/sampled_image_filenames.txt".format(name)
        output_dynamic_mask_path = "/data3/VSRD_PP_Sync/est_dynamic_list/sync{}/dynamic_mask.txt".format(name)
        
        
        process_the_data(dynamic_mask_list_path=dynamic_mask_list_path,
                        sample_group_list_path=sample_group_list_path,
                        output_path=output_dynamic_mask_path                     
                        )

    

        
        
        
        
        
        

