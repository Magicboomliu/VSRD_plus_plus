import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch


def load_pickle(file_path):
    """
    Load a pickle file and return its content.
    
    Args:
        file_path (str): Path to the pickle file.
    
    Returns:
        object: The content of the pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except pickle.UnpicklingError:
        print(f"Error: The file at {file_path} is not a valid pickle file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__=="__main__":
    
    # file_path = "/home/zliu/TPAMI25/AutoLabels/sdflabel/test_labels/demo.pkl"
    # data = load_pickle(file_path)
    # print(data[1])
    
    kitti_sample = torch.load("/home/zliu/TPAMI25/AutoLabels/SDFlabel/data/optimization/kitti_sample.pt")
    
    print(kitti_sample['lidar'].max())
    print(kitti_sample['lidar'].min())
    
    print(kitti_sample.keys())
    
    print(kitti_sample['world_to_cam'])
    print(kitti_sample['orig_cam'])
    
    print(kitti_sample['depth'].max())
    
    

    a = torch.zeros(4,4)
    a[:3,:3] = torch.eye(3)
    a[3,3]=1
    a[0,3] = 5.97421356e-02
    a[1,3] = -3.57286467e-04
    a[2,3] = 2.74096891e-03
    print(a)