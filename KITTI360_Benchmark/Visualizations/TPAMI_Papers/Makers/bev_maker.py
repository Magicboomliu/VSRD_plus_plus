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

import cv2

from scipy.optimize import linear_sum_assignment

def compute_iou_matrix(boxes_A, boxes_B):
    """
    """
    N1, N2 = len(boxes_A), len(boxes_B)
    iou_matrix = np.zeros((N1, N2))

    for i, box_a in enumerate(boxes_A):
        for j, box_b in enumerate(boxes_B):
            x1 = max(box_a[0], box_b[0])
            y1 = max(box_a[1], box_b[1])
            x2 = min(box_a[2], box_b[2])
            y2 = min(box_a[3], box_b[3])

            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            union_area = area_a + area_b - inter_area + 1e-6

            iou_matrix[i, j] = inter_area / union_area

    return iou_matrix

def bilateral_matching_linear_sum_assignment(boxes_A, boxes_B):
    """
    """
    # 计算 IoU 矩阵
    iou_matrix = compute_iou_matrix(boxes_A, boxes_B)

    # 将 IoU 转化为损失矩阵（负值或反向变换）
    cost_matrix = 1 - iou_matrix

    # 使用 linear_sum_assignment 求解最小成本匹配
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # 筛选有效的匹配对（IoU > 0 的部分）
    matched_index_A = []
    matched_index_B = []
    for i, j in zip(row_indices, col_indices):
        if iou_matrix[i, j] > 0:  # 忽略 IoU 为 0 的匹配
            matched_index_A.append(i)
            matched_index_B.append(j)

    return matched_index_A, matched_index_B

def select_with_index_from_list(old_list,index_list):
    
    new_list = []
    for my_index in index_list:
        new_list.append(old_list[my_index])
    
    return new_list
                
def read_annotation(annotation_filename,class_names=['car']):

    with open(annotation_filename) as file:
        annotation = json.load(file)

    intrinsic_matrix = torch.as_tensor(annotation["intrinsic_matrix"])
    extrinsic_matrix = torch.as_tensor(annotation["extrinsic_matrix"])
    
    
    instance_ids = {
        class_name: list(masks.keys())
        for class_name, masks in annotation["masks"].items()
        if class_name in class_names
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


    return dict(
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=extrinsic_matrix,
    )

def read_image(image_path):
    image_data = np.array(Image.open(image_path))
    return image_data


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def extract_from_2013(path):
    if "2013" in path:
        result = path.split("2013", 1)[1]
        return "2013" + result
    else:
        return None

def start_with_2013(string):
    return string[string.index("2013"):]


def instance_to_segmentation(instance_mask):
    """
    Converts an instance mask of shape [N, H, W] into a whole segmentation mask of shape [H, W].

    Args:
        instance_mask (np.ndarray): Instance mask of shape [N, H, W].

    Returns:
        np.ndarray: Whole segmentation mask of shape [H, W].
    """
    # Get the number of instances (N) and the spatial dimensions (H, W)
    N, H, W = instance_mask.shape

    # Initialize the whole segmentation mask
    segmentation_mask = np.zeros((H, W), dtype=np.int32)

    # Assign each instance a unique label (starting from 1)
    for i in range(N):
        segmentation_mask[instance_mask[i] > 0] =1.0  # Assume non-zero pixels belong to the instance

    return segmentation_mask

def instance_to_colored_segmentation(instance_mask):
    """
    Converts an instance mask of shape [N, H, W] into a colored segmentation mask of shape [H, W, 3].

    Args:
        instance_mask (np.ndarray): Instance mask of shape [N, H, W].

    Returns:
        np.ndarray: Colored segmentation mask of shape [H, W, 3].
    """
    N, H, W = instance_mask.shape

    # Generate the whole segmentation mask
    segmentation_mask = np.zeros((H, W), dtype=np.int32)
    for i in range(N):
        segmentation_mask[instance_mask[i] > 0] = i + 1  # Assign unique labels to instances

    # Map unique labels to distinct colors
    unique_labels = np.unique(segmentation_mask)
    color_map = plt.cm.get_cmap('tab20', len(unique_labels))  # Choose a colormap (tab20 has distinct colors)
    colors = (color_map(np.arange(len(unique_labels)))[:, :3] * 255).astype(np.uint8)  # RGB colors

    # Create the colored mask
    colored_segmentation = np.zeros((H, W, 3), dtype=np.uint8)
    for label, color in zip(unique_labels, colors):
        if label == 0:  # Skip background
            continue
        colored_segmentation[segmentation_mask == label] = color

    return colored_segmentation

def masks_to_bboxes(masks, min_bbox_size=1):
    """
    Convert instance masks to bounding boxes ensuring that all boxes are valid.
    
    Args:
        masks (np.ndarray): A numpy array of shape [N, H, W] where N is the number of instances,
                            H is the height, and W is the width of the masks.
        min_bbox_size (int): The minimum size of the bounding box sides. Default is 1.
    
    Returns:
        np.ndarray: A numpy array of shape [N, 4] where each row contains [x_min, y_min, x_max, y_max]
                    coordinates of the bounding box for each instance. Ensures that x_max > x_min
                    and y_max > y_min.
    """
    num_instances = masks.shape[0]
    bboxes = np.zeros((num_instances, 4), dtype=np.int32)
    
    for idx in range(num_instances):
        # Find the rows and columns that contain the mask
        rows, cols = np.where(masks[idx] > 0)
        
        if len(rows) == 0 or len(cols) == 0:
            # If no mask is found, set bbox to zero or some default invalid values
            bboxes[idx] = [0, 0, 0, 0]
        else:
            x_min = np.min(cols)
            x_max = np.max(cols)
            y_min = np.min(rows)
            y_max = np.max(rows)
            
            # Ensure the bounding box has minimum size
            if x_max <= x_min + min_bbox_size - 1:
                x_max = x_min + min_bbox_size - 1
            if y_max <= y_min + min_bbox_size - 1:
                y_max = y_min + min_bbox_size - 1
            
            # Store the bounding box
            bboxes[idx] = [x_min, y_min, x_max, y_max]
    
    return bboxes

def convert_into_dynamic_mask(instance_mask,dynamic_mask):
    """
    Converts an instance mask of shape [N, H, W] into a whole segmentation mask of shape [H, W].

    Args:
        instance_mask (np.ndarray): Instance mask of shape [N, H, W].

    Returns:
        np.ndarray: Whole segmentation mask of shape [H, W].
    """
    # Get the number of instances (N) and the spatial dimensions (H, W)
    N, H, W = instance_mask.shape

    # Initialize the whole segmentation mask
    segmentation_mask = np.zeros((H, W), dtype=np.int32)

    # Assign each instance a unique label (starting from 1)
    for i in range(N):
        is_dyamic = dynamic_mask[i]
        
        segmentation_mask[instance_mask[i] > 0] =1.0 * is_dyamic  # Assume non-zero pixels belong to the instance

    return segmentation_mask

def draw_boxes_bev(image, boxes_3d, extents=((-40.0, 80.0), (40.0, 0.0)),
                                            color=None,thickness=None,lineType=None):
    is_float = image.dtype.kind == "f"

    if is_float:
        image = skimage.img_as_ubyte(image)
    image = np.ascontiguousarray(image)

    # NOTE: use the KITTI-360 "evaluation" format instaed of the KITTI-360 "annotation" format
    # NOTE: the KITTI-360 "annotation" format is different from the KITTI-360 "evaluation" format
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/prepare_train_val_windows.py#L133
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L552

    boxes_2d = np.mean(boxes_3d.reshape(-1, 2, 4, 3), axis=1)
    boxes_2d = (boxes_2d[..., [0, 2]] - extents[0]) / -np.subtract(*extents) * image.shape[:2][::-1]
    
    for box_2d in boxes_2d:

        for (point_1, point_2) in zip(box_2d, np.roll(box_2d, shift=-1, axis=0)):
            image = cv2.line(
                img=image,
                pt1=tuple(map(int, point_1)),
                pt2=tuple(map(int, point_2)),
                color=color,
                thickness= thickness,
                lineType= lineType
                
            )

    for y in range(0, image.shape[0], image.shape[0] // 10):

        image = cv2.line(
            img=image,
            pt1=(0, y),
            pt2=(image.shape[1], y),
            color=(128, 128, 128),
        )

    for x in range(0, image.shape[1], image.shape[1] // 10):

        image = cv2.line(
            img=image,
            pt1=(x, 0),
            pt2=(x, image.shape[0]),
            color=(128, 128, 128),
        )



    if is_float:
        image = skimage.img_as_float32(image)

    return image


def main(root_folder,
         pd_folder,
         output_folder):
    
    os.makedirs(output_folder,exist_ok=True)
    sequence_name_list = sorted(os.listdir(pd_folder))
    
    # BEV
    output_bev_folder = os.path.join(output_folder,"bev_folder")
    os.makedirs(output_bev_folder,exist_ok=True)
    

    for idx, seq_name in enumerate(sequence_name_list):
        print("current sequence name {}/{}".format(idx+1,len(sequence_name_list)))
        pd_labels_folder_of_current_seq = os.path.join(pd_folder,seq_name,'image_00/data_rect')
        for fname in tqdm(sorted(os.listdir(pd_labels_folder_of_current_seq))):
            # estimated labels of the vsrd_plus_plus
            estimated_labels_path = os.path.join(pd_labels_folder_of_current_seq,fname)
            assert os.path.exists(estimated_labels_path)
            # gt_labels
            gt_labels_path = estimated_labels_path.replace("predictions","my_gts_with_dynamic")
            assert os.path.exists(gt_labels_path)
            # image 0
            image_00_path = os.path.join(root_folder,"data_2d_raw",seq_name,'image_00/data_rect',fname.replace(".txt",".png"))
            assert os.path.exists(image_00_path)
            # annotations
            annotations_path = os.path.join(root_folder,"annotations",seq_name,'image_00/data_rect',fname.replace(".txt",".json"))
            assert os.path.exists(annotations_path)
            # calib path
            calib_path = os.path.join(root_folder,"cam_calib.txt")
            assert os.path.exists(calib_path)

            # camera extrinsic / intrinsic/ rectification matrix
            annotions_raw_data = read_annotation(annotations_path,class_names=['car'])
            intrinsic_matrix = torch.as_tensor(annotions_raw_data["intrinsic_matrix"])
            extrinsic_matrix = torch.as_tensor(annotions_raw_data["extrinsic_matrix"])
            target_extrinsic_matrix = extrinsic_matrix
            inverse_target_extrinsic_matrix = torch.linalg.inv(target_extrinsic_matrix)

            x_axis, y_axis, _ = target_extrinsic_matrix[..., :3, :3]
            rectification_angle = (
                torch.acos(torch.dot(torch.round(y_axis), y_axis)) *
                torch.sign(torch.dot(torch.cross(torch.round(y_axis), y_axis), x_axis))
            )
            rectification_matrix = rotation_matrix_x(rectification_angle).cpu().numpy()
            
            if "masks" in annotions_raw_data.keys():
                gt_mask = annotions_raw_data['masks'] # (N,H,W)
            else:
                continue
            
            # gt_mask shape is [N,H,W]
            gt_2d_bounding_box_from_mask = masks_to_bboxes(gt_mask)
            
            # image data path
            image_data = read_image(image_00_path)
            # gt 3D bounding boxes
            gt_object_data = get_objects_from_label(gt_labels_path)
            # estimated 3D bounding boxes
            est_obj_data = get_objects_from_label(estimated_labels_path)
            gt_boxes2d_data = np.array([obj.box2d for obj in gt_object_data])
            
            dynamic_mask_initial = np.loadtxt(gt_labels_path,dtype=str).reshape(-1,17)[:,-1]
            dynamic_mask_initial = dynamic_mask_initial.astype(np.float32)
            

        
            est_boxes2d_data = np.array([obj.box2d for obj in est_obj_data])
            
            gt_matched_index, est_matched_index = bilateral_matching_linear_sum_assignment(boxes_A=gt_boxes2d_data,
                                                                                         boxes_B=est_boxes2d_data)
        
            gt_object_data = select_with_index_from_list(old_list=gt_object_data,
                                                         index_list=gt_matched_index)
            est_obj_data = select_with_index_from_list(old_list=est_obj_data,
                                                       index_list=est_matched_index)
            
            dynamic_mask_initial = dynamic_mask_initial[gt_matched_index]
                                    
            
            calib_data = get_calib_from_file(calib_path)

            # saved bev folders
            copied_image_for_bev = (np.ones((1000,1000,3))*255).astype(np.uint8)
            saved_basename = start_with_2013(image_00_path)[:-4]
            
            
            for inner_idx, object in enumerate(est_obj_data):
                if inner_idx<len(gt_object_data):
                    gt_corners3d = gt_object_data[inner_idx].generate_corners3d_raw(rectification_matrix)
                    gt_corners2d = project_to_image(gt_corners3d,calib_data["P2"])
                    if gt_corners2d.min()<-10:
                        continue
                    # Estimated Labels
                    est_corners3d = est_obj_data[inner_idx].generate_corners3d_raw(rectification_matrix)
                    projected_est_corners3d = project_to_image(est_corners3d,calib_data["P2"])
                    if projected_est_corners3d.min()<-0:
                        continue

                    if np.isnan(copied_image_for_bev).any():
                        continue
                    copied_image_for_bev = draw_boxes_bev(image=copied_image_for_bev,
                                            color=(255, 0, 0),
                                            thickness=2,
                                            extents=((-50.0, 100.0), (50.0, 0.0)),
                                            lineType=cv2.LINE_AA,
                                            boxes_3d=gt_corners3d)
                    if np.isnan(copied_image_for_bev).any():
                        continue

                    copied_image_for_bev = draw_boxes_bev( 
                                    image=copied_image_for_bev,
                                    color=(0, 0, 255),
                                    thickness=2,
                                    extents=((-50.0, 100.0), (50.0, 0.0)),
                                    lineType=cv2.LINE_AA,
                                    boxes_3d=est_corners3d)
                    if np.isnan(copied_image_for_bev).any():
                        continue

    
            
            copied_image_for_bev= copied_image_for_bev.astype(np.uint8)

            
            saved_bev_fname = os.path.join(output_bev_folder,saved_basename+".png")

            if not os.path.exists(saved_bev_fname):
                os.makedirs(os.path.dirname(saved_bev_fname),exist_ok=True)
                skimage.io.imsave(saved_bev_fname,(copied_image_for_bev).astype(np.uint8))




if __name__=="__main__":
    
    # root folder 
    root_folder= "/media/zliu/data12/dataset/VSRD_PP_Sync/"
    # estimated labels
    pd_folder = "/media/zliu/data12/TPAMI_Results/Stage1_Expermental_Results/VSRD_PP_Completed/pseudo_labels_txt/predictions/"
    # output folders
    output_folder = "/media/zliu/data12/TPAMI_Results/Figures_For_Papers/VSRD_PP_Results_All/"

    main(root_folder=root_folder,
                pd_folder=pd_folder,
                output_folder=output_folder)
