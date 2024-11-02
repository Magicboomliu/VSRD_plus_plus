import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pickle
import sys
from PIL import Image
import open3d as o3d


LINE_INDICES = [[0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],]


def read_pickle_file(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def get_depth_filename(image_filename):
    annotation_filename = (
        image_filename
        .replace("data_2d_raw", "IGEVStereoSSL")
    )
    return annotation_filename

def read_depth(filename):
    depth = np.array(Image.open(filename))
    depth = depth.astype(np.float32) / 256.
    return depth

def save_to_pickle(pickle_file_path,saved_dict):
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(saved_dict, f)
        
   
def read_image_tensor(image_path,device="cuda:0"):
    image = np.array(Image.open(image_path).convert("RGB")).astype(np.float32)/255.
    image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(device)
    return image

def merge_for_visualization_version_1(pcds):
    raw_pcd_set = []
    for key in pcds.keys():
        if pcds[key] is not None:
            raw_pcd_set.append(pcds[key])
            
    return raw_pcd_set


def visualize_point_cloud_with_axis(point_cloud,axis_vis=True,boxes_3d=None):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    if axis_vis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True) 
    vis.add_geometry(pcd)
    
    if boxes_3d is not None:
        for box in boxes_3d:
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(box)
            lineset.lines = o3d.utility.Vector2iVector(LINE_INDICES)
            colors = [[0, 1, 0] for _ in range(len(LINE_INDICES))]  
            lineset.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(lineset)
    
    
    vis.add_geometry(axis)
    vis.run() 
    vis.destroy_window() 
    

def visualize_point_cloud_with_axis_with_two_box(axis_vis=True,boxes_3d1=None,boxes_3d2=None):

    if axis_vis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True) 

    
    if boxes_3d1 is not None:
        for box in boxes_3d1:
            lineset = o3d.geometry.LineSet()

            lineset.points = o3d.utility.Vector3dVector(box)
            lineset.lines = o3d.utility.Vector2iVector(LINE_INDICES)
            colors = [[0, 1, 0] for _ in range(len(LINE_INDICES))]  
            lineset.colors = o3d.utility.Vector3dVector(colors)

            vis.add_geometry(lineset)

    if boxes_3d2 is not None:
        for box in boxes_3d2:
            lineset2 = o3d.geometry.LineSet()

            lineset2.points = o3d.utility.Vector3dVector(box)
            lineset2.lines = o3d.utility.Vector2iVector(LINE_INDICES)
            colors = [[1, 0, 0] for _ in range(len(LINE_INDICES))]  
            lineset2.colors = o3d.utility.Vector3dVector(colors)

            vis.add_geometry(lineset2)
    
    
    vis.add_geometry(axis)
    
    vis.run() 
    vis.destroy_window() 
    
    


def visualize_point_cloud_with_axis_with_two_box_with_pcd(point_cloud,axis_vis=True,boxes_3d1=None,boxes_3d2=None):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    if axis_vis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True) 
    vis.add_geometry(pcd)
    
    
    if boxes_3d1 is not None:
        for box in boxes_3d1:
            lineset = o3d.geometry.LineSet()

            lineset.points = o3d.utility.Vector3dVector(box)
            lineset.lines = o3d.utility.Vector2iVector(LINE_INDICES)
            colors = [[0, 1, 0] for _ in range(len(LINE_INDICES))]  
            lineset.colors = o3d.utility.Vector3dVector(colors)

            vis.add_geometry(lineset)


    if boxes_3d2 is not None:
        for box in boxes_3d2:
            lineset2 = o3d.geometry.LineSet()

            lineset2.points = o3d.utility.Vector3dVector(box)
            lineset2.lines = o3d.utility.Vector2iVector(LINE_INDICES)
            colors = [[1, 0, 0] for _ in range(len(LINE_INDICES))]  
            lineset2.colors = o3d.utility.Vector3dVector(colors)

            vis.add_geometry(lineset2)
    
    
    vis.add_geometry(axis)
    
    vis.run() 
    vis.destroy_window() 