import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import sys


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def cut_with_beginning_of_2013(string):
    return string[string.index("2013"):]



if __name__=="__main__":
    
    root_dirname = "/media/zliu/data12/dataset/VSRD_PP_Sync/"
    split_dirname = "R50-N16-M128-B16"
    sequnces = sorted(os.listdir("/media/zliu/data12/dataset/Tsubame_Results/V1/ckpts/data_2d_raw/"))
    dynamic_dirname = "/media/zliu/data12/dataset/VSRD_PP_Sync/est_dynamic_list/"
    ckpt_dirname = "/media/zliu/data12/dataset/Tsubame_Results/V1/ckpts/"
    ckpt_filename = "step_2499.pt"
    saved_psuedo_folder = "predictions"
    
    static_dynamic_training_splits = "/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrdpp_splist/original_ImageSets/ImageSets/train.txt"
    static_dynamic_training_filenames = read_text_lines(static_dynamic_training_splits)
    


    valid_ckpt_name_list = []
    for idx, sequence in enumerate(sequnces): 
        # group txt
        group_filename = os.path.join(root_dirname, "filenames", split_dirname, sequence, "grouped_image_filenames.txt")
        assert os.path.exists(group_filename)
        with open(group_filename) as file:
            grouped_image_filenames = {
                tuple(map(int, line.split(" ")[0].split(","))): line.split(" ")[1].split(",")
                for line in map(str.strip, file)}
            
    

        # sample txt
        sample_filename = os.path.join(root_dirname, "filenames", split_dirname, sequence, "sampled_image_filenames.txt")
        assert os.path.exists(sample_filename)
        with open(sample_filename) as file:
            sampled_image_filenames = {
                tuple(map(int, line.split(" ")[0].split(","))): line.split(" ")[1]
                for line in map(str.strip, file)
            }
        
        # dynamic txt
        dynamic_txt_filename = os.path.join(dynamic_dirname,"sync"+sequence[-7:-5],"dynamic_mask.txt")
        assert os.path.exists(dynamic_txt_filename)
        with open(dynamic_txt_filename) as file:
            dynamic_instance_list = {
                tuple(map(int, line.split(" ")[0].split(","))): [int(float(item)) for item in line.split(" ")[2].split(",")] 
                for line in map(str.strip, file)
            }
            
        
        
        for instance_ids, grouped_image_filenames in grouped_image_filenames.items():

            # get the target image filename
            target_image_filename = sampled_image_filenames[instance_ids]
            # get the instance dynamic list
            instance_dynamic_list = dynamic_instance_list[instance_ids]
            # image direction filenames
            target_image_dirname = os.path.splitext(os.path.relpath(target_image_filename, root_dirname))[0]
            # get the models ckpts
            target_ckpt_filename = os.path.join(ckpt_dirname, target_image_dirname, ckpt_filename)

            if not os.path.exists(target_ckpt_filename):
                print(f"[{target_ckpt_filename}] Does not exist!")
                continue

            assert os.path.exists(target_ckpt_filename)
            assert os.path.exists(target_image_filename)
            
        

            for source_image_filename in grouped_image_filenames:

                source_annotation_filename = source_image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")
                assert os.path.exists(source_annotation_filename)
                source_prediction_dirname = os.path.join(saved_psuedo_folder, os.path.basename(ckpt_dirname))
                source_prediction_filename = source_annotation_filename.replace("annotations", source_prediction_dirname)
                
                source_prediction_filename_for_searching = os.path.join("data_2d_raw",cut_with_beginning_of_2013(source_annotation_filename).replace(".json",'.png'))
                
                if source_prediction_filename_for_searching in static_dynamic_training_filenames:
                    if target_ckpt_filename not in valid_ckpt_name_list:
                        valid_ckpt_name_list.append(target_ckpt_filename)
                


    with open("ablation_studies_sequence_name.txt",'w')  as f:
        for idx, line in enumerate(valid_ckpt_name_list):
            if idx!=len(valid_ckpt_name_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)

