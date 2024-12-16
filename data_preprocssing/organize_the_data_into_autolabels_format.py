import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import json
import pycocotools.mask
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def save_to_pt_file(file_path, data_dict):
    """
    Save a dictionary to a .pt file (not zip-like, just like the provided format).

    Args:
        file_path (str): Path to save the .pt file.
        data_dict (dict): The dictionary to save.
    """
    # Save the dictionary to a .pt file
    torch.save(data_dict, file_path)

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def read_annotation(annotation_filename ,class_names=['car']):

    with open(annotation_filename) as file:
        annotation = json.load(file)

    intrinsic_matrix = torch.as_tensor(annotation["intrinsic_matrix"])
    extrinsic_matrix = torch.as_tensor(annotation["extrinsic_matrix"])

    instance_ids = {
        class_name: list(masks.keys())
        for class_name, masks in annotation["masks"].items()
        if class_name in ['car']
    }

    if instance_ids:

        masks = torch.cat([
            torch.as_tensor(np.stack([
                pycocotools.mask.decode(annotation["masks"][class_name][instance_id])
                for instance_id in instance_ids
            ]), dtype=torch.float)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0) #(N,H,W)

        labels = torch.cat([
            torch.as_tensor([class_names.index(class_name)] * len(instance_ids), dtype=torch.long)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0) # class

        boxes_3d = torch.cat([
            torch.as_tensor([
                annotation["boxes_3d"][class_name].get(instance_id, [[np.nan] * 3] * 8)
                for instance_id in instance_ids
            ], dtype=torch.float)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        instance_ids = torch.cat([
            torch.as_tensor(list(map(int, instance_ids)), dtype=torch.long)
            for instance_ids in instance_ids.values()
        ], dim=0)

        return dict(
            masks=masks,
            labels=labels,
            boxes_3d=boxes_3d,
            instance_ids=instance_ids,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
        )

    else:

        return dict(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
        )

def depth_to_pointcloud(depth, K, extrinsic):
    """
    Convert a sparse depth map into a 3D point cloud in the world coordinate system.
    
    Args:
        depth (numpy.ndarray): Sparse depth map of shape (H, W).
        K (numpy.ndarray): Camera intrinsics matrix of shape (3, 3).
        extrinsic (numpy.ndarray): Extrinsic matrix (4, 4), world-to-camera transformation.
    
    Returns:
        numpy.ndarray: Point cloud in world coordinates of shape (N, 3).
    """
    # Get image dimensions
    H, W = depth.shape

    # Create a meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x[depth > 0]  # Filter valid depth points
    y = y[depth > 0]
    z = depth[depth > 0]

    # Create homogeneous image coordinates (u, v, 1)
    uv1 = np.stack([x, y, np.ones_like(x)], axis=0)

    # Compute camera coordinates (Xc, Yc, Zc) using intrinsics
    K_inv = np.linalg.inv(K)
    cam_coords = (K_inv @ (uv1 * z)).T  # Shape: (N, 3)

    # Convert to homogeneous camera coordinates (Xc, Yc, Zc, 1)
    cam_coords_h = np.concatenate([cam_coords, np.ones((cam_coords.shape[0], 1))], axis=1)

    # Transform to world coordinates
    extrinsic_inv = np.linalg.inv(extrinsic)
    world_coords_h = (extrinsic_inv @ cam_coords_h.T).T  # Shape: (N, 4)

    # Remove the homogeneous dimension
    world_coords = world_coords_h[:, :3]

    return world_coords

def compute_bounding_boxes_float(instances):
    """
    Compute bounding boxes for each instance in a tensor and retain float coordinates.
    
    Args:
        instances (torch.Tensor): Input tensor of shape (N, H, W), 
                                  where N is the number of instances.
    
    Returns:
        torch.Tensor: Bounding boxes of shape (N, 4), where each box is 
                      (x_min, y_min, x_max, y_max) as float values.
    """
    N, H, W = instances.shape
    bounding_boxes = torch.zeros((N, 4), dtype=torch.float32)  # (x_min, y_min, x_max, y_max)

    for n in range(N):
        # Get the binary mask for the nth instance
        mask = instances[n] > 0  # Assumes non-zero values indicate presence

        if mask.any():  # If the instance has valid pixels
            # Find the indices of the non-zero pixels
            y_coords, x_coords = torch.nonzero(mask, as_tuple=True)

            # Compute bounding box using float precision
            x_min, x_max = x_coords.float().min().item(), x_coords.float().max().item()
            y_min, y_max = y_coords.float().min().item(), y_coords.float().max().item()

            # Store in the result tensor
            bounding_boxes[n] = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        else:
            # If the mask is empty, leave the bounding box as (0, 0, 0, 0)
            bounding_boxes[n] = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    return bounding_boxes







def create_kitti360_data_for_autolabels_sample(scale,
                                               idx,
                                               name,
                                               lidar_path,
                                               annotations_file_path,
                                               cam_calib_path,
                                               sync_gt_label_path,
                                               sync_image_path):
    
    data_sample_dict = dict()
    data_sample_dict['idx'] = idx
    data_sample_dict['scale'] = scale
    data_sample_dict['name'] = name
    projected_depth = np.load(lidar_path)

    # image = np.array(Image.open(sync_image_path).convert("RGB"))/255.
    image = cv2.imread(sync_image_path)/255.
    data_sample_dict['image'] = image
    data_sample_dict['orig_hw'] = image.shape
    
    K_old = np.array([float(data) for data in read_text_lines(cam_calib_path)[2].split()[1:]]).reshape(3,4)
    K = K_old[:3,:3]
    
    
    
    
    annotation_contents = read_annotation(annotations_file_path)
    world_to_cam = annotation_contents['extrinsic_matrix']

    cam, R, t = cv2.decomposeProjectionMatrix(K_old)[:3]
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = R
    world_to_cam[:3, 3] = -t[:3, 0]
    
    data_sample_dict['world_to_cam'] = world_to_cam
    
    data_sample_dict['orig_cam'] = cam

    lidar = depth_to_pointcloud(depth=projected_depth,K=K,extrinsic=world_to_cam) + 8
    
    
    data_sample_dict['lidar'] = lidar
    data_sample_dict['depth'] = projected_depth 
    
    
    gt_annotations = np.loadtxt(sync_gt_label_path,dtype=str).reshape(-1,17)
    
    gt_dict_list = []
    annos = dict()
    annos['easy'] = []
    annos['medium'] = []
    annos['hard'] = []
    
    for idx, line in enumerate(gt_annotations):
        gt_dict = dict()
        gt_dict['name'] = line[0]
        gt_dict['bbox'] = [float(a) for a in line[4:8].tolist()]
        gt_dict['location'] =[float(a) for a in line[11:14].tolist()]
        gt_dict['dimensions'] =[float(a) for a in line[8:11].tolist()]
        gt_dict['rotation_y'] = float(line[14])
        gt_dict['score'] = float(line[15])
        gt_dict['truncated'] = 0.0
        gt_dict['occluded'] = 0
        gt_dict['ignore'] = False
        gt_dict['alpha'] = float(line[3])
        gt_dict_list.append(gt_dict)
    
    data_sample_dict['gt'] = gt_dict_list
    annos['easy'] = gt_dict_list
    data_sample_dict['annos'] = annos   
    
    
    maskrcnn_dict = dict()
    
    bboxes = compute_bounding_boxes_float(annotation_contents['masks'])
    
    mask_list = []
    for idx, sample in enumerate(bboxes):
        
        height_length = bboxes[idx][3] - bboxes[idx][1]
        width_length = bboxes[idx][2] - bboxes[idx][0]
        
        if height_length ==0:
            bboxes[idx][3] = bboxes[idx][3] + 1
            
        if width_length ==0:
            bboxes[idx][2] = bboxes[idx][2] + 1
        
        mask = annotation_contents['masks'][idx] #[H,W]
        mask = mask[int(bboxes[idx][1]):int(bboxes[idx][3]),int(bboxes[idx][0]):int(bboxes[idx][2])]
        mask_list.append(mask)
    
    
    maskrcnn_dict['bboxes'] = bboxes
    maskrcnn_dict['masks'] = mask_list
        
    
    return data_sample_dict,maskrcnn_dict
    
    

def create_kitti360_data_for_autolabels_sample_V2(scale,
                                               idx,
                                               name,
                                               lidar_path,
                                               annotations_file_path,
                                               cam_calib_path,
                                               sync_gt_label_path,
                                               sync_image_path):
    
    data_sample_dict = dict()
    data_sample_dict['idx'] = idx
    data_sample_dict['scale'] = scale
    data_sample_dict['name'] = name
    projected_depth = np.load(lidar_path)

    # image = np.array(Image.open(sync_image_path).convert("RGB"))/255.
    image = cv2.imread(sync_image_path)/255.
    data_sample_dict['image'] = image
    data_sample_dict['orig_hw'] = image.shape
    
    K_old = np.array([float(data) for data in read_text_lines(cam_calib_path)[2].split()[1:]]).reshape(3,4)
    K = K_old[:3,:3]
    
    
    
    
    annotation_contents = read_annotation(annotations_file_path)
    world_to_cam = annotation_contents['extrinsic_matrix']

    cam, R, t = cv2.decomposeProjectionMatrix(K_old)[:3]
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = R
    world_to_cam[:3, 3] = -t[:3, 0]
    
    data_sample_dict['world_to_cam'] = world_to_cam
    
    data_sample_dict['orig_cam'] = cam

    lidar = depth_to_pointcloud(depth=projected_depth,K=K,extrinsic=world_to_cam) + 8
    
    
    data_sample_dict['lidar'] = lidar
    data_sample_dict['depth'] = projected_depth 
    
    
    gt_annotations = np.loadtxt(sync_gt_label_path,dtype=str).reshape(-1,17)
    
    gt_dict_list = []
    annos = dict()
    annos['easy'] = []
    annos['medium'] = []
    annos['hard'] = []
    
    for idx, line in enumerate(gt_annotations):
        gt_dict = dict()
        gt_dict['name'] = line[0]
        gt_dict['bbox'] = [float(a) for a in line[4:8].tolist()]
        gt_dict['location'] =[float(a) for a in line[11:14].tolist()]
        gt_dict['dimensions'] =[float(a) for a in line[8:11].tolist()]
        gt_dict['rotation_y'] = float(line[14])
        gt_dict['score'] = float(line[15])
        gt_dict['truncated'] = 0.0
        gt_dict['occluded'] = 0
        gt_dict['ignore'] = False
        gt_dict['alpha'] = float(line[3])
        gt_dict_list.append(gt_dict)
    
    data_sample_dict['gt'] = gt_dict_list
    annos['easy'] = gt_dict_list
    data_sample_dict['annos'] = annos   
    
    
    maskrcnn_dict = dict()
    
    bboxes = compute_bounding_boxes_float(annotation_contents['masks'])
    
    mask_list = []
    for idx, sample in enumerate(bboxes):
        
        height_length = bboxes[idx][3] - bboxes[idx][1]
        width_length = bboxes[idx][2] - bboxes[idx][0]
        
        if height_length ==0:
            bboxes[idx][3] = bboxes[idx][3] + 1
            
        if width_length ==0:
            bboxes[idx][2] = bboxes[idx][2] + 1
        
        mask = annotation_contents['masks'][idx] #[H,W]
        mask = mask[int(bboxes[idx][1]):int(bboxes[idx][3]),int(bboxes[idx][0]):int(bboxes[idx][2])]
        mask_list.append(mask)
    
    
    maskrcnn_dict['bboxes'] = bboxes
    maskrcnn_dict['masks'] = mask_list
        
    
    return data_sample_dict,maskrcnn_dict









# generate all the lists
if __name__=="__main__":

    
    image_2_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000251.png"
    scale = 1
    idx = 1
    name = "data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000251"
    lidar_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/projected_lidar/2013_05_28_drive_0000_sync/projected_lidar/data/0000000251.npy"
    annotations_file_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/annotations/2013_05_28_drive_0000_sync/image_00/data_rect/0000000251.json"
    cam_calib_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/cam_calib.txt"
    
    synced_image_2_path = '/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT/training/image_2/000001.png'
    synced_label_gt_path = synced_image_2_path.replace("image_2","label_gt").replace(".png",'.txt')
    
    # det2d_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/det2d/threshold03/2013_05_28_drive_0000_sync/image_00/data_rect/0000000251.txt"
    
    assert os.path.exists(synced_image_2_path)
    assert os.path.exists(synced_label_gt_path)
    
    # kitti360_sample_data_dict,maskrnn_dict = create_kitti360_data_for_autolabels_sample(scale=scale,
    #                                            idx=idx,
    #                                            name=name,
    #                                            lidar_path=lidar_path,
    #                                            annotations_file_path=annotations_file_path,
    #                                            cam_calib_path=cam_calib_path,
    #                                            sync_gt_label_path=synced_label_gt_path,
    #                                            sync_image_path=synced_image_2_path)

    kitti360_sample_data_dict,maskrnn_dict = create_kitti360_data_for_autolabels_sample_V2(scale=scale,
                                               idx=idx,
                                               name=name,
                                               lidar_path=lidar_path,
                                               annotations_file_path=annotations_file_path,
                                               cam_calib_path=cam_calib_path,
                                               sync_gt_label_path=synced_label_gt_path,
                                               sync_image_path=synced_image_2_path)


        

    # save_to_pt_file(file_path="/home/zliu/TPAMI25/AutoLabels/SDFlabel/data/KITTI360_Example/0000000251.pt",
    #                 data_dict=kitti360_sample_data_dict)

    # save_to_pt_file(file_path="/home/zliu/TPAMI25/AutoLabels/SDFlabel/data/KITTI360_Example/maskrcnn_0000000251.pt",
    #                 data_dict=maskrnn_dict)
    
    
    
    
    
    pass