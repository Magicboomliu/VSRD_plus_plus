import os
import pickle

# 读取 .pkl 文件
def read_pkl_file(filename):
    with open(filename, 'rb') as f:  # 'rb' 表示以二进制方式读取文件
        data = pickle.load(f)  # 反序列化数据
    return data





if __name__=="__main__":
    
    pkl_path = "/home/zliu/Downloads/pre_gen_kitti_raw_data_lidar_RoI_points/kitti_raw/2011_09_26/2011_09_26_drive_0015_sync/lidar_RoI_points/data/0000000000.pkl"

    pkl_data = read_pkl_file(pkl_path)
    
    print(pkl_data['RoI_points'][1].shape) # [nums_of_lidars,3]
    
    
    print(pkl_data['bbox2d'].shape)