import os
import numpy as np
import open3d as o3d
import cv2
from torch.utils.data import Dataset
from collections import OrderedDict
cv2.setNumThreads(0)
import sys
sys.path.append("..")
# from utils.refinement import is_anno_easy, is_anno_moderate, compute_depth_map, reproject, build_view_frustum
from utils.refinement import is_anno_easy_kitti360, is_anno_hard_kitti360,compute_depth_map, reproject, build_view_frustum
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt



def get_kitti_frame(sample):
    H, W, _ = sample['image'].shape
    # Filter out lidar points outside field of view
    scene_lidar = sample['lidar']
    frustum = build_view_frustum(sample['orig_cam'], 0, 0, W, H)
    scene_lidar = scene_lidar[np.logical_and.reduce(frustum @ scene_lidar.T > 0, axis=0)]

    # Build Open3D pcd and estimate normals
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_lidar)
    scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    # origin_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)

    # Estimate road plane (stupidly by checking orthogonality to camera. RANSAC better)
    normals = np.asarray(scene_pcd.normals)
    ortho_to_cam = np.abs(normals @ np.asarray([0, 1, 0])) > 0.9
    plane_points = scene_lidar[ortho_to_cam]
    plane_normal = np.mean(normals[ortho_to_cam], axis=0)
    plane_normal /= np.linalg.norm(plane_normal)
    plane_dists = plane_normal @ plane_points.T
    plane_offset = np.median(plane_dists)

    # Filter out road plane by simple normal check
    scene_lidar = scene_lidar[~ortho_to_cam]
    scene_pcd.points = o3d.utility.Vector3dVector(scene_lidar)
    scene_pcd.normals = o3d.utility.Vector3dVector(normals[~ortho_to_cam])
    # o3d.visualization.draw_geometries([scene_pcd, origin_mesh])

    # Compute depth map for whole image
    scene_depth = compute_depth_map(scene_lidar, sample['orig_cam'], W, H)

    # Reproject all visible, colored scene points
    pts_scene, clrs_scene = reproject(sample['image'], scene_depth, sample['orig_cam'])
    pcd = o3d.geometry.PointCloud()
    pcd.points, pcd.colors = o3d.utility.Vector3dVector(pts_scene), o3d.utility.Vector3dVector(clrs_scene)
    return scene_depth, pcd


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



def start_with_2013(string):
    return string[string.index("2013_"):]




def compute_velodyne_to_camera(camera_to_velodyne):
    """
    计算 velodyne_to_camera 矩阵 (3x4)。
    
    参数:
        camera_to_velodyne (np.ndarray): 3x4 矩阵，表示从 camera 到 velodyne 的转换。
        
    返回:
        velodyne_to_camera (np.ndarray): 3x4 矩阵，表示从 velodyne 到 camera 的转换。
    """
    if camera_to_velodyne.shape != (3, 4):
        raise ValueError("Input matrix must be a 3x4 matrix.")
    
    # 扩展成4x4齐次矩阵
    camera_to_velodyne_homogeneous = np.eye(4)
    camera_to_velodyne_homogeneous[:3, :] = camera_to_velodyne
    
    # 计算逆矩阵
    velodyne_to_camera_homogeneous = np.linalg.inv(camera_to_velodyne_homogeneous)
    
    # 提取前3x4部分
    velodyne_to_camera = velodyne_to_camera_homogeneous[:3, :]
    
    return velodyne_to_camera




def blend_depth_with_rgb_masked(depth_map, rgb_image, alpha=0.1, colormap=cv2.COLORMAP_JET):
    """
    将深度图和RGB图融合，深度为0的区域保留原始RGB图。
    
    参数:
        depth_map (np.ndarray): 深度图 (H, W)，值为离散深度。
        rgb_image (np.ndarray): RGB图像 (H, W, 3)，范围为 [0, 255]。
        alpha (float): Alpha混合系数，范围 [0, 1]。
        colormap (int): 用于深度图的伪彩色映射，默认使用 cv2.COLORMAP_JET。
    
    返回:
        blended_image (np.ndarray): 融合后的图像，范围 [0, 255]。
    """
    if depth_map.shape != rgb_image.shape[:2]:
        raise ValueError("Depth map and RGB image dimensions must match.")
    
    # 归一化深度图到 [0, 255]，深度为0的地方保持为0
    normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 使用伪彩色映射深度图
    depth_colored = cv2.applyColorMap(normalized_depth, colormap)
    
    # 创建深度掩码 (mask)，深度为0的地方掩盖
    depth_mask = (depth_map > 0).astype(np.uint8)  # 深度不为0的地方为1，为0的地方为0
    
    # 将RGB图像从uint8转换为float32，以进行混合
    rgb_image_float = rgb_image.astype(np.float32)
    depth_colored_float = depth_colored.astype(np.float32)
    
    # 对深度有效的地方进行Alpha Blending
    blended_image = rgb_image_float.copy()
    blended_image[depth_mask == 1] = cv2.addWeighted(
        rgb_image_float[depth_mask == 1],
        alpha,
        depth_colored_float[depth_mask == 1],
        1 - alpha,
        0
    )
    
    # 转换回uint8
    blended_image = blended_image.astype(np.uint8)
    
    return blended_image


class KITTI360(Dataset):
    def __init__(
        self,
        root_path,
        syned_root_path,
        filename_list_path = None):

        self.root_path = root_path
        self.syned_root_path = syned_root_path
        self.filename_list_path = filename_list_path
        
        # read training file
        awaited_preprocessed_contents = read_text_lines(self.filename_list_path)
        
        
        self.names = []
        self.images = []
        self.label_files = []
        self.calibs = []
        self.lidars = []
        
        self.cam_to_velos = []
        
        
                
        
        for line in awaited_preprocessed_contents:
            splits = line.split()
            true_filename_path, synced_filename_path = splits
            
            current_name = start_with_2013(true_filename_path)[:-4]
            
            label_name = synced_filename_path.replace("image_2","label_gt").replace(".png",'.txt')
            calib_name = synced_filename_path.replace("image_2","calib").replace(".png",".txt")
            lidar_name = true_filename_path.replace("data_2d_raw","data_3d_raw").replace("image_00/data_rect/","velodyne_points/data/").replace(".png",".bin")
            lidar_name = os.path.join(self.root_path,lidar_name)
            cam_to_velo_name = os.path.join(self.root_path,"calibration/calib_cam_to_velo.txt")
            
            assert os.path.exists(calib_name)
            assert os.path.exists(label_name)
            assert os.path.exists(lidar_name)
            assert os.path.exists(cam_to_velo_name)
            
            self.names.append(current_name)
            self.images.append(synced_filename_path)
            self.label_files.append(label_name)
            self.calibs.append(calib_name)
            self.lidars.append(lidar_name)
            self.cam_to_velos.append(cam_to_velo_name)
        
        
        
        
            

        # depth_prefix = 'lidar_depth'
        # os.makedirs(os.path.join(path, 'lidar_depth'), exist_ok=True)

        # if data_split == 'test':
        #     self.depths = [depth_prefix + '/test_' + name + '.npz' for name in self.names]
        # else:
        #     self.depths = [depth_prefix + '/train_' + name + '.npz' for name in self.names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # Start building sample
        sample = OrderedDict()
        sample['idx'] = index
        sample['scale'] = 1
        sample['name'] = self.names[index]
        

        

        # Read calibration data and break apart into separate entities
        calib = open(self.calibs[index]).readlines()
        calib = [c[:-1].split(' ') for c in calib]

        # Parse left camera projection matrix into 3x4 form
        P2 = np.asarray([float(f) for f in calib[2][1:]]).reshape((3, 4))
        

        # get the velo_to_cam matrix
        
        cam_to_velo_raw = open(self.cam_to_velos[index])
        cam_to_velo_raw = [c.split() for c in cam_to_velo_raw]
        cam_to_velo_raw = np.asarray([float(f) for f in cam_to_velo_raw[0]]).reshape((3, 4)) # 3x4
        
        # Parse velodyne to left image transform
        velo_to_cam = compute_velodyne_to_camera(camera_to_velodyne=cam_to_velo_raw)


        # Reshape LIDAR data into (x, y, z, intensity) and bring into camera frame
        velodyne = np.fromfile(self.lidars[index], np.float32)
        velodyne = velodyne.reshape((-1, 4))[:, :3]
        sample['lidar'] = (velo_to_cam[:3, :3] @ velodyne.T).T + velo_to_cam[:3, 3]


        # Read the image and label files
        img = cv2.imread(self.images[index], -1)
        H, W, C = img.shape
        sample['image'] = img.astype(np.float32) / 255.0
        sample['orig_hw'] = (H, W)
        
        # Decompose projection matrix into 'cam' intrinsics and rotation
        cam, R, t = cv2.decomposeProjectionMatrix(P2)[:3]

        # Store original intrinsics
        sample['orig_cam'] = cam.copy()

        # NOTE: We should multiply with world_to_cam, but difference is small
        sample['world_to_cam'] = np.eye(4)
        sample['world_to_cam'][:3, :3] = R
        sample['world_to_cam'][:3, 3] = -t[:3, 0]

        # Load depth map in meters
        # depth_url = os.path.join(self.path, self.depths[index])

        # Break labels apart into separate entities (only if needed... slow!)
        labels = open(self.label_files[index]).readlines()
        sample['gt'] = []
        for label in [l[:-1].split(' ') for l in labels]:
            trunc = float(label[1])  # From 0 to 1 (truncated), how much object leaving image
            occ = int(label[2])  # 0 = visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
            alpha = float(label[3])  # Observation angle of object, ranging [-pi..pi]
            dimensions = [float(b) for b in label[8:11]]  # height, width, length (in meters)
            location = [float(b) for b in label[11:14]]  # 3D (ground) location in camera
            rot_y = float(label[14])  # Rotation around Y-axis in camera [-pi..pi]

            anno = {}
            anno['name'] = label[0]  # Describes the type of object: 'Car', 'Van', 'Truck'
            anno['bbox'] = [int(float(b)) for b in label[4:8]]  # LTRB in pixels
            anno['location'] = location
            anno['dimensions'] = dimensions
            anno['rotation_y'] = rot_y
            anno['alpha'] = alpha
            anno['score'] = 1
            anno['truncated'] = trunc
            anno['occluded'] = occ

            # Throw away all 3D information for unlabeled 3D boxes (set to -1000)
            anno['ignore'] = location[0] < -100

            sample['gt'].append(anno)

        # Some car instances are completely occluded by other things because annotated with LIDAR
        if True:
            for inst_i, anno_i in enumerate(sample['gt']):
                for inst_j, anno_j in enumerate(sample['gt']):
                    if anno_i['name'] != 'Car' or inst_j == inst_i:
                        continue

                    # Compute Intersection normalized by anno_i's area
                    # Measures how much the box is subsumed by the other
                    inter_lt = np.maximum(anno_i['bbox'][:2], anno_j['bbox'][:2])
                    inter_br = np.minimum(anno_i['bbox'][2:], anno_j['bbox'][2:])
                    inter_wh = np.maximum(inter_br - inter_lt, 0)
                    intersection = (inter_wh[0] * inter_wh[1]) / ((anno_i['bbox'][2] - anno_i['bbox'][0]) *
                                                                    (anno_i['bbox'][3] - anno_i['bbox'][1]))

                    # Some 'DontCare's were simply put over other annotations... Jesus...
                    if intersection > 0.5 and anno_j['name'] == 'DontCare':
                        anno_i['ignore'] = True

                    # Check if 2D bbox fully inside another 2D bbox but Z larger, deactivate if true
                    if True:
                        if not anno_i['ignore'] and not anno_j['ignore']:
                            if anno_i['location'][2] > anno_j['location'][2] and intersection > 0.95:
                                anno_i['ignore'] = True
                                break

        # Filter all valid Car annotations based on type and difficulty
        annos = {'easy': [], 'medium': [], 'hard': []}
        for anno in sample['gt']:
            if anno['name'] != 'Car' or anno['ignore']:
                continue
            if is_anno_easy_kitti360(anno):
                annos['easy'].append(anno)
            elif is_anno_hard_kitti360(anno):
                annos['hard'].append(anno)
                

        depth, pcd = get_kitti_frame(sample)
        

        sample['depth'] = depth
        sample['pcd'] = pcd
        sample['annos'] = annos

        return sample




if __name__=="__main__":

    root_path = "/media/zliu/data12/dataset/VSRD_PP_Sync/"
    syned_root_path = "/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT"
    processed_filenames_all = "/home/zliu/TPAMI25/AutoLabels/SDFlabel/data_preprocssing/all_filenames.txt"

    kitti_360_dataset = KITTI360(root_path=root_path,
                                 syned_root_path= syned_root_path,
                                 filename_list_path=processed_filenames_all)
    
    
    for idx, sample in enumerate(kitti_360_dataset):
        print(sample.keys())
        
        quit()
        
    
    