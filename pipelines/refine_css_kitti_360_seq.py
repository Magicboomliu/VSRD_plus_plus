import os
import math
from collections import defaultdict, OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import pickle

import sys
sys.path.append("..")

import utils.data as config
from networks.resnet_css import setup_css
from datasets.kitti360 import KITTI360
from pipelines.detection_3d import Detection3DEvaluator, clean_kitti_data, CoordinateFrame
from pipelines.optimizer import Optimizer
from sdfrenderer.grid import Grid3D
from utils.pose import PoseEstimator
import sdfrenderer.deepsdf.workspace as dsdf_ws
import utils.refinement as rtools
import utils.visualizer as viztools

# Define seed for reproducibility
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
from tqdm import tqdm

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
    synced_kitti_path = config.INPUT.SYNCED_KITTI_PATH
    # filename
    filename_list = config.INPUT.FILENAME
    label_type = config.INPUT.LABEL_TYPE # gt
    kitti = KITTI360(root_path=root_path,syned_root_path=synced_kitti_path,
                     filename_list_path=filename_list)

    # Throw in here all the annotations and labels we acquire over all frames
    total_annotations, total_estimations = OrderedDict(), OrderedDict()    
    idx = 0
    
    for sample in tqdm(kitti):
        name = sample['name']
        saved_pd_labels = os.path.join(path_autolabels,name) +".txt"
        
        os.makedirs(os.path.dirname(saved_pd_labels),exist_ok=True)
        
        if os.path.exists(saved_pd_labels):
            print("File Exists Already!")
            continue
        
        if not [a for a in sample['gt'] if a['name'] == 'Car']:
            continue

        # Build container dicts to hold annotations and labels for later evaluation
        frame_annos, frame_estimations = defaultdict(list), defaultdict(list)

        diff_annos = config.INPUT.DIFF_ANNOS
        annos = rtools.get_annos_kitti360s(diff_annos,sample)
        
        # Loop through annotations
        for anno_idx, anno in enumerate(annos):
            # Store this annotation for later evaluation
            [frame_annos[key].append(value) for key, value in anno.items()]
            
            max_crop_area = 1024

            # Get detected crop
            l, t, r, b = anno['bbox']
            crop_bgr = sample['image'][t:b, l:r].copy()
            crop_dep = sample['depth'][t:b, l:r].copy()


            # Adjust intrinsics based on crop position and size
            K = sample['orig_cam']
            crop_size = torch.Tensor(crop_bgr.shape[:-1])
            crop_size, intrinsics, off_intrinsics = rtools.adjust_intrinsics_crop(
                K, crop_size, anno['bbox'], max_crop_area
            )

            pcd_crop, pcd_crop_rgb = rtools.reproject(crop_bgr, crop_dep, off_intrinsics, filter=False)

            # Preprocess image patch for pytorch digestion
            crop_rgb, crop_rgb_vis = rtools.transform_bgr_crop(crop_bgr, orig=True)
            crop_rgb = crop_rgb.unsqueeze(0).to(device).float()

            # Get css output
            pred_css = css_net(crop_rgb)
            nocs_pred = pred_css['uvw_sm_masked'].detach().squeeze() / 255.
            latent_pred = pred_css['latent'][0].detach().to(precision)

            # DeepSDF Inference and surface point/normal extraction.
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
            pose_esimator_type = 'kabsch'
            scale = 2.0  # scale_dsdf.item() * 0.8
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
                        
            viz_type = None

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

            label_kitti, scaled_points, cam_T = rtools.get_kitti_label(dsdf, grid, params['latent'].to(precision),
                                                       params['scale'].to(precision), params['trans'].to(precision),
                                                       params['yaw'].to(precision), sample['world_to_cam'], anno['bbox'])
            [frame_estimations[key].append(value) for key, value in label_kitti.items()]

        # Do not store anything for this frame if we did not process any annotations
        if not frame_annos:
            continue

        # Transform all annotations and labels into needed format and save these frame results
        necessary_keys = ['alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'score']
        for key in necessary_keys:
            frame_annos[key] = np.asarray(frame_annos[key])
            frame_estimations[key] = np.asarray(frame_estimations[key])
        

        save_the_results_into_kitti3d_format(frame_estimations,save_name=saved_pd_labels)
