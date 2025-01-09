import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pycocotools.mask
import torch
import json
import skimage.io
import argparse
from tqdm import tqdm
import sys
sys.path.append("..")
from draw_projected_3d import draw3d_bbox_2d_projection,draw3d_bbox_2d_projection_V2
from kitti_box_computation import project_to_image
from kitti_utils import get_calib_from_file, get_objects_from_label
from geo_op import rotation_matrix_x,rotation_matrix_y,rotation_matrix_z
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines




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
        
        
        for instance_ids, grouped_image_filenames in grouped_image_filenames.items():

            # get the target image filename
            target_image_filename = sampled_image_filenames[instance_ids]
            # image direction filenames
            target_image_dirname = os.path.splitext(os.path.relpath(target_image_filename, root_dirname))[0]
            # get the models ckpts
            target_ckpt_filename = os.path.join(ckpt_dirname, target_image_dirname, ckpt_filename)

            if not os.path.exists(target_ckpt_filename):
                print(f"[{target_ckpt_filename}] Does not exist!")
                continue

            assert os.path.exists(target_ckpt_filename)
            assert os.path.exists(target_image_filename)
            
            # Record the Velocity
            
            
            
            print(target_image_filename)
            print("************************************************")
            for source_image_filename in grouped_image_filenames:

                print(source_image_filename)
            
            print("---------------------------------")
