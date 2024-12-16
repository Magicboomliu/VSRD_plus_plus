import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config_refine_kitti_360 import _C as config
from pipelines.refine_css_kitti_360_seq import refine_css_kitti_360
import argparse 


if __name__=="__main__":
    
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str,default="")

    # Parse arguments
    args = parser.parse_args()
    
    config.INPUT.FILENAME = args.filename
    
    refine_css_kitti_360(config)
    
    pass