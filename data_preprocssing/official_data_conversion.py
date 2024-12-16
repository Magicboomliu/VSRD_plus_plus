import torch
import numpy as np
import torch.nn.functional as F
import os



# generate all the lists
if __name__=="__main__":

    
    image_2_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000251.png"
    scale = 1
    idx = 1
    name = "data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000251"
    lidar_path = "/media/zliu/data12/dataset/KITTI/KITTI360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000251.bin"
    annotations_file_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/annotations/2013_05_28_drive_0000_sync/image_00/data_rect/0000000251.json"
    cam_calib_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/cam_calib.txt"
    
    synced_image_2_path = '/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT/training/image_2/000001.png'
    synced_label_gt_path = synced_image_2_path.replace("image_2","label_gt").replace(".png",'.txt')
    
    # det2d_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/det2d/threshold03/2013_05_28_drive_0000_sync/image_00/data_rect/0000000251.txt"
    
    assert os.path.exists(synced_image_2_path)
    assert os.path.exists(synced_label_gt_path)
    assert os.path.exists(lidar_path)