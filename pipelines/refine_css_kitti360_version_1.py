import os
import math
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

import utils.data as config
from networks.resnet_css import setup_css
from pipelines.optimizer import Optimizer
from sdfrenderer.grid import Grid3D
from utils.pose import PoseEstimator
import sdfrenderer.deepsdf.workspace as dsdf_ws
import utils.refinement as rtools
import utils.visualizer as viztools
import matplotlib.pyplot as plt
import pickle

import json
import pycocotools.mask
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from tqdm import tqdm


# Define seed for reproducibility
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


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
    
    K = np.array([float(data) for data in read_text_lines(cam_calib_path)[2].split()[1:]]).reshape(3,4)[:3,:3]
    data_sample_dict['orig_cam'] = K
    
    annotation_contents = read_annotation(annotations_file_path)
    world_to_cam = annotation_contents['extrinsic_matrix'].cpu().numpy()

    # default settings of the 
    a = torch.zeros(4,4)
    a[:3,:3] = torch.eye(3)
    a[3,3]=1
    a[0,3] = 5.97421356e-02
    a[1,3] = -3.57286467e-04
    a[2,3] = 2.74096891e-03
    
    world_to_cam = a    
    data_sample_dict['world_to_cam'] = world_to_cam
    

    
    lidar = depth_to_pointcloud(depth=projected_depth,K=K,extrinsic=world_to_cam)
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
    
    
def save_the_results_into_kitti3d_format(dict,save_name):
    
    names_list = dict['name']
    line_list = []
    for idx, name in enumerate(names_list):
        type_name = dict['name'][idx]
        truncated = 0.0
        occluded = 0.0
        alpha = dict['alpha'].tolist()[idx]
        box = dict['bbox'][idx].tolist()
        box1 = box[0]
        box2 = box[1]
        box3 = box[2]
        box4 = box[3]
        dimensions = dict['dimensions'][idx].tolist()
        dim1 = dimensions[0]
        dim2 = dimensions[1]
        dim3 = dimensions[2]
        locations = dict['location'][idx].tolist()
        loc1 = locations[0]
        loc2 = locations[1]
        loc3 = locations[2]
        rotation_y = dict["rotation_y"].tolist()[idx]
        score = dict['score'].tolist()[idx]
        line = type_name + " " + str(truncated) + " " + str(occluded) + " "+ str(alpha) + " " + str(box1) + " " + str(box2) + " " +str(box3) + " "+ str(box4) + " " + str(dim1) + " " + str(dim2) + " "+ str(dim3) + " "+ str(loc1)+ " "+ str(loc2) + " " + str(loc3) + " "+ str(rotation_y)+" " + str(score)
        line_list.append(line)
    
    
    with open(save_name,'w') as f:
        for idx, line in enumerate(line_list):
            if idx!=len(line_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)


def start_with_2013_(string):
    return string[string.index("2013"):]


def refine_css_kitti_360(config):
    
    # Set device and precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = config.OPTIMIZATION.PRECISION
    if precision=="float16":
        precision = torch.float16


    # Setup CSS
    css_path = config.INPUT.CSS_PATH
    css_net = setup_css(pretrained=True, model_path=css_path).to(device)
    

    # Setup DeepSDF
    dsdf_path = config.INPUT.DEEPSDF_PATH
    dsdf, latent_size = dsdf_ws.setup_dsdf(dsdf_path, precision=precision)
    dsdf = dsdf.to(device)

    # Define label type (GT, MaskRCNN)
    label_type = config.INPUT.LABEL_TYPE
    

    # Path for output autolabels
    path_autolabels = config.LABELS.PATH
    os.makedirs(path_autolabels, exist_ok=True)


    # ROOT PATH
    root_path = config.INPUT.KITTI_PATH
    # filename
    filename_list = config.INPUT.FILENAME
    
    awaited_preprocessed_contents = read_text_lines(filename_list)

    for line in tqdm(awaited_preprocessed_contents):
        splits = line.split()
        image_2_path = splits[0]
        scale = 1.0
        idx = 1.0
        name = image_2_path[:-4]
        
        basename = os.path.basename(image_2_path).replace(".png",'.txt')
        saved_folder_name = os.path.dirname(start_with_2013_(image_2_path))
        saved_folder_name = os.path.join(path_autolabels,saved_folder_name)
        os.makedirs(saved_folder_name,exist_ok=True)
        
        saved_filename = os.path.join(saved_folder_name,basename)
        
        if os.path.exists(saved_filename):
            continue
    

        
        sparse_depth_path = image_2_path.replace("data_2d_raw","sparse_depth").replace("image_00","projected_lidar").replace(".png",".npy").replace("data_rect","data")
        sparse_depth_path = os.path.join(root_path,sparse_depth_path)
        image_2_path_abs = os.path.join(root_path,image_2_path)
        annotations_file_path = image_2_path.replace("data_2d_raw","annotations").replace(".png",".json")
        annotations_file_path = os.path.join(root_path,annotations_file_path)
        cam_calib_path = os.path.join(root_path,"cam_calib.txt")
        
        synced_image_2_path = splits[1]
        synced_label_gt_path = synced_image_2_path.replace("image_2","label_gt").replace(".png",'.txt')
        
        
        assert os.path.exists(sparse_depth_path)
        assert os.path.exists(image_2_path_abs)
        assert os.path.exists(annotations_file_path)
        assert os.path.exists(synced_image_2_path)
        assert os.path.exists(cam_calib_path)
        assert os.path.exists(synced_label_gt_path)

        kitti360_sample_data_dict,maskrnn_dict = create_kitti360_data_for_autolabels_sample(scale=scale,
                                                idx=idx,
                                                name=name,
                                                lidar_path=sparse_depth_path,
                                                annotations_file_path=annotations_file_path,
                                                cam_calib_path=cam_calib_path,
                                                sync_gt_label_path=synced_label_gt_path,
                                                sync_image_path=synced_image_2_path)
        
        
        sample = kitti360_sample_data_dict

        # Build container dicts to hold annotations and labels for later evaluation
        frame_annos, frame_estimations = defaultdict(list), defaultdict(list)

        # Select annotations based on difficulty
        diff_annos = config.INPUT.DIFF_ANNOS
        annos = rtools.get_annos(diff_annos, sample)

        # Load MaskRCNN labels, skip frame if no labels found
        if label_type != 'gt':
            maskrcnn_labels = maskrnn_dict

        
        # Loop through annotations
        for anno_idx, anno in enumerate(annos):
            
            # Store this annotation for later evaluation
            [frame_annos[key].append(value) for key, value in anno.items()]

            # If maskrcnn labels are available
            if label_type != 'gt':
                # Find closest maskrcnn bbox by iou
                iou = []
                for id, bbox in enumerate(maskrcnn_labels['bboxes'].numpy()):
                    iou.append(rtools.get_iou(bbox, anno['bbox']))

                bbox_max_id = np.argmax(iou)
                bbox_maskrcnn = maskrcnn_labels['bboxes'][bbox_max_id].numpy()
                anno['bbox'] = bbox_maskrcnn.astype(np.int64)
                

            # Get crops
            #max_crop_area = config.read_cfg_int(cfgp, 'input', 'rendering_area', default=64) ** 2
            max_crop_area = 1024

            # Get detected crop
            l, t, r, b = anno['bbox']
            crop_bgr = sample['image'][t:b, l:r].copy() # get image batch
            crop_dep = sample['depth'][t:b, l:r].copy() # get the depth patch

            # Adjust intrinsics based on crop position and size
            K = sample['orig_cam']  # camera intrinsics
            crop_size = torch.Tensor(crop_bgr.shape[:-1])
            crop_size, intrinsics, off_intrinsics = rtools.adjust_intrinsics_crop(
                K, crop_size, anno['bbox'], max_crop_area
            )  # 裁剪后的相机内参矩阵, 考虑偏移量的内参矩阵，用于点云重投影。
            
            # (N,3)
            pcd_crop, pcd_crop_rgb = rtools.reproject(crop_bgr, crop_dep, off_intrinsics, filter=False) # 生成裁剪区域的点云


            # Use masks from maskrcnn
            if label_type == 'maskrcnn':
                mask = maskrcnn_labels['masks'][bbox_max_id]
                crop_bgr *= mask.unsqueeze(-1).float().expand_as(torch.tensor(crop_bgr)).numpy()

            # Preprocess image patch for pytorch digestion
            crop_rgb, crop_rgb_vis = rtools.transform_bgr_crop(crop_bgr, orig=True)
            crop_rgb = crop_rgb.unsqueeze(0).to(device).float()

            # Get css output
            pred_css = css_net(crop_rgb)
            nocs_pred = pred_css['uvw_sm_masked'].detach().squeeze() / 255.
            latent_pred = pred_css['latent'][0].detach().to(precision)

            # DeepSDF Inference and surface point/normal extraction.
            #grid_density = config.read_cfg_int(cfgp, 'input', 'grid_density', default=30)
            grid_density = 40
            grid = Grid3D(grid_density, device, precision)

            inputs = torch.cat([latent_pred.expand(grid.points.size(0), -1), grid.points],
                            1).to(latent_pred.device, latent_pred.dtype)
            pred_sdf_grid, inv_scale = dsdf(inputs)
            pcd_dsdf, nocs_dsdf, normals_dsdf = grid.get_surface_points(pred_sdf_grid)

            # Reproject NOCS into the scene
            nocs_pred_resized = F.interpolate(nocs_pred.unsqueeze(0), size=crop_dep.shape[:2],
                                            mode='nearest').squeeze(0)
            nocs_3d_pts, nocs_3d_cls = rtools.reproject(
                nocs_pred_resized, torch.Tensor(crop_dep).unsqueeze(0), off_intrinsics, filter=True
            )

            # Estimating initial pose
            #pose_esimator_type = config.read_cfg_string(cfgp, 'optimization', 'pose_estimator', default='kabsch')
            pose_esimator_type = 'kabsch'
            scale = 2.0
            pose_esimator = PoseEstimator(pose_esimator_type, scale)
            init_pose = pose_esimator.estimate(
                pcd_dsdf, nocs_dsdf, nocs_3d_pts, nocs_3d_cls, off_intrinsics, nocs_pred_resized
            )

            if init_pose is None:
                print('NO RANSAC POSE FOUND!!!')
                continue
            scale, rot, tra = init_pose['scale'], init_pose['rot'], init_pose['tra']

            # Constrain rotation to azimuth only. We need to flip X from the car system
            rot[:, 1] = [0, 1, 0]
            rot[1, :] = [0, 1, 0]
            yaw = rtools.roty_in_bev(rot @ np.diag([-1, 1, 1])) + math.pi / 2  # KITTI roty starts at canonical pi/2

            # Estimate good height by looking up lowest Y value of reprojected NOCS
            world_points = ((rot @ (pcd_dsdf.detach().cpu().numpy() * scale).T).T + tra)
            proj_world = rtools.project(sample['orig_cam'], world_points)
            L, T = proj_world[:, 0].min(), proj_world[:, 1].min()
            R, B = proj_world[:, 0].max(), proj_world[:, 1].max()
            iou = rtools.compute_iou([l, t, r, b], [L, T, R, B])
            if iou < 0.7:
                print('Restimating height')
                ymin, ymax = world_points[:, 1].min(), world_points[:, 1].max()
                tra[1] = nocs_3d_pts[:, 1].min() + (ymax - ymin) / 2

            # Optimizer and params
            params = {}
            params['yaw'] = np.array([yaw])
            params['trans'] = init_pose['tra'] / init_pose['scale']
            params['scale'] = np.array([init_pose['scale']])
            params['latent'] = latent_pred.detach().cpu().numpy()

            weights = {}
            # weights['2d'] = config.read_cfg_float(cfgp, 'losses', '2d_weight', default=1)
            # weights['3d'] = config.read_cfg_float(cfgp, 'losses', '3d_weight', default=1)
            weights['2d'] = 0.3
            weights['3d'] = 0.5

            # Refine the initial estimate
            optimizer = Optimizer(params, device, weights)

            # For additional visualization
            frame_vis = {}
            frame_vis['image'] = sample['image']
            frame_vis['bbox'] = anno['bbox']
            frame_vis['crop_size'] = crop_bgr.shape[:-1]

            # Set visualization type
            #viz_type = config.read_cfg_string(cfgp, 'visualization', 'viz_type', default=None)
            viz_type = '3d'
            viz_type = None

            # Optimize the initial pose estimate
            #iters_optim = config.read_cfg_int(cfgp, 'optimization', 'iters', default=100)
            iters_optim = 60
            optimizer.optimize(
                iters_optim,
                nocs_pred,
                pcd_crop,
                dsdf,
                grid,
                intrinsics.detach().to(device, precision),
                crop_size,
                frame_vis=frame_vis,
                viz_type=viz_type
            )

            # Now collect the results from the optimization
            label_kitti, scaled_points, cam_T = rtools.get_kitti_label(dsdf, grid, params['latent'].to(precision),
                                                    params['scale'].to(precision), params['trans'].to(precision),
                                                    params['yaw'].to(precision), sample['world_to_cam'], anno['bbox'])
            [frame_estimations[key].append(value) for key, value in label_kitti.items()]
            # viztools.plot_3d_final(sample['lidar'], cam_T, scaled_points, anno, label_kitti)

        # Transform all annotations and labels into needed format and save these frame results
        necessary_keys = ['alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'score']
        for key in necessary_keys:
            frame_annos[key] = np.asarray(frame_annos[key])
            frame_estimations[key] = np.asarray(frame_estimations[key])
        
        
       
        save_the_results_into_kitti3d_format(dict=frame_estimations,save_name=saved_filename)
 
        
    
    
