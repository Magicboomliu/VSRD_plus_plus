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



import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
 
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_blue_to_red_cmap():
    # 定义从深蓝到深红的颜色映射
    colors = [
        '#0000ff',  # 深蓝
        '#007fff',  # 天蓝
        '#00ffff',  # 青色
        '#00ff7f',  # 春绿
        '#7fff00',  # 黄绿
        '#ffff00',  # 黄色
        '#ff7f00',  # 橙色
        '#ff0000'   # 深红
    ]
    cmap = LinearSegmentedColormap.from_list("custom_blue_to_red", colors, N=256)
    return cmap

def visualize_instance_masks_on_image_and_save(background_image, masks, confidences, file_path):
    background_image = background_image.astype(np.float32) / 255
    masks = masks.cpu().numpy()
    assert masks.shape[0] == len(confidences), "Number of masks must match the number of confidence scores"
    H, W = masks.shape[1], masks.shape[2]
    norm = Normalize(vmin=0, vmax=1)
    cmap = create_blue_to_red_cmap()

    # Initialize an image for the mask overlay
    visualization = np.copy(background_image)  # Start with a copy of the background image

    # Iterate over each mask and its corresponding confidence
    for i in range(masks.shape[0]):
        mask = masks[i].astype(bool)  # Ensure mask is boolean
        confidence = confidences[i]
        color = np.array(cmap(norm(confidence))[:3])  # Get RGB color from the colormap

        # Update visualization only where mask is true
        for c in range(3):
            # Properly broadcasting the mask to all color channels
            visualization[mask, c] = (1 - 0.4) * visualization[mask, c] + 0.4 * color[c]

    # Clip the visualization to ensure values are within a valid range
    visualization = np.clip(visualization, 0, 1)


    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(visualization)
    ax.axis('off')

    # Create an axis for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)

    # Create a colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)

    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)




def main(root_folder,pd_folder,output_folder):
    os.makedirs(output_folder,exist_ok=True)
    sequence_name_list = sorted(os.listdir(pd_folder))
    
    # projected 3d only
    output_folder_projected_3D_only_path = os.path.join(output_folder,"projected_est_3d_box_only")
    os.makedirs(output_folder_projected_3D_only_path,exist_ok=True)
    
    # projected 3d and GT Only
    output_folder_projected_3D_And_GT_path = os.path.join(output_folder,'projected_gt_and_est_boxes')
    os.makedirs(output_folder_projected_3D_And_GT_path,exist_ok=True)
    
    # projected GT Only
    output_folder_gt_only = os.path.join(output_folder,"projected_gt_boxes_only")
    os.makedirs(output_folder_gt_only,exist_ok=True)
    
    
    # BEV
    output_bev_folder = os.path.join(output_folder,"bev_folder")
    os.makedirs(output_bev_folder,exist_ok=True)
    
    # Segmentation Mask
    output_seg_mask_path = os.path.join(output_folder,'seg_mask')
    os.makedirs(output_seg_mask_path,exist_ok=True)
    
    # Instance Mask
    output_instance_mask_path = os.path.join(output_folder,'instance_mask')
    os.makedirs(output_instance_mask_path,exist_ok=True)
    
    
    # dynamic Mask
    output_dynamic_mask = os.path.join(output_folder,'dynamic_mask')
    os.makedirs(output_dynamic_mask,exist_ok=True)
    
    
    # Confidence Mask
    output_confidnce_mask = os.path.join(output_folder,'confidence_mask')
    os.makedirs(output_confidnce_mask,exist_ok=True)
    
    # confidnce Mask with the Predicted Boxes
    output_confidnce_mask_with_pd = os.path.join(output_folder,'confidence_mask_with_pd')
    os.makedirs(output_confidnce_mask_with_pd,exist_ok=True)
    
    
    
    
    for idx, seq_name in enumerate(sequence_name_list):
        
        #FIXME
        if seq_name!="2013_05_28_drive_0010_sync":
            continue

        
        print("current sequence name {}/{}".format(idx+1,len(sequence_name_list)))
        pd_labels_folder_of_current_seq = os.path.join(pd_folder,seq_name,'image_00/data_rect')
        for fname in tqdm(sorted(os.listdir(pd_labels_folder_of_current_seq))):
            #FIXME
            if fname!='0000001677.txt':
                continue

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
            
            confidence_mask_initial = np.loadtxt(estimated_labels_path,dtype=str).reshape(-1,16)[:,-1]
            confidence_mask_initial = confidence_mask_initial.astype(np.float32)
            
        
            
            est_boxes2d_data = np.array([obj.box2d for obj in est_obj_data])
            
            gt_matched_index, est_matched_index = bilateral_matching_linear_sum_assignment(boxes_A=gt_boxes2d_data,
                                                                                         boxes_B=est_boxes2d_data)
        
            gt_object_data = select_with_index_from_list(old_list=gt_object_data,
                                                         index_list=gt_matched_index)
            est_obj_data = select_with_index_from_list(old_list=est_obj_data,
                                                       index_list=est_matched_index)
            
            #FIXME
            # dynamic_mask_initial = dynamic_mask_initial[gt_matched_index]
            confidence_mask_initial = confidence_mask_initial[est_matched_index]
            
            
                                    
            
            calib_data = get_calib_from_file(calib_path)
            
            
            # projected 3D BOXES
            copied_image_for_gt_labels = image_data.copy()
            copied_image_for_estimate = image_data.copy()
            copied_image_for_gt_and_est = image_data.copy()
 

            for inner_idx, object in enumerate(est_obj_data):
                if inner_idx<len(gt_object_data):
                    gt_corners3d = gt_object_data[inner_idx].generate_corners3d_raw(rectification_matrix)
                    gt_corners2d = project_to_image(gt_corners3d,calib_data["P2"])
                    if gt_corners2d.min()<-10:
                        continue

                    if np.isnan(copied_image_for_gt_labels).any():
                        continue
                    if np.isnan(copied_image_for_estimate).any():
                        continue
                    if np.isnan(copied_image_for_gt_and_est).any():
                        continue
                            
                
                    
                    # GT Labels
                    copied_image_wth_gt_labels = draw3d_bbox_2d_projection_V2(copied_image_for_gt_labels,qs=gt_corners2d,color=(0,255,0),is_gt=True)
                    if np.isnan(copied_image_wth_gt_labels).any():
                        copied_image_wth_gt_labels = copied_image_for_gt_labels
                    else:
                        copied_image_for_gt_labels = copied_image_wth_gt_labels
                    
                    # Estimated Labels
                    est_corners3d = est_obj_data[inner_idx].generate_corners3d_raw(rectification_matrix)
                    projected_est_corners3d = project_to_image(est_corners3d,calib_data["P2"])
                    if projected_est_corners3d.min()<-0:
                        continue
                    
                    
                    copied_image_with_est_labels = draw3d_bbox_2d_projection_V2(copied_image_for_estimate,qs=projected_est_corners3d,color=(0,255,0),is_gt=False)
                    if copied_image_with_est_labels is  None:
                        copied_image_with_est_labels = copied_image_for_estimate
                    else:
                        copied_image_for_estimate = copied_image_with_est_labels

                    # Estimated Labels With GT Labels
    
                    copied_image_with_est_and_gt_labels = draw3d_bbox_2d_projection_V2(copied_image_for_gt_and_est,qs=projected_est_corners3d,color=(0,255,0),is_gt=False)
                    if np.isnan(copied_image_with_est_and_gt_labels).any():
                        copied_image_with_est_and_gt_labels = copied_image_for_gt_and_est
                    else:
                        copied_image_for_gt_and_est = copied_image_with_est_and_gt_labels

                    copied_image_with_est_and_gt_labels = draw3d_bbox_2d_projection_V2(copied_image_for_gt_and_est,qs=gt_corners2d,color=(0,255,0),is_gt=True)
                    if np.isnan(copied_image_with_est_and_gt_labels).any():
                        copied_image_with_est_and_gt_labels = copied_image_for_gt_and_est
                    else:
                        copied_image_for_gt_and_est = copied_image_with_est_and_gt_labels
                    
            
            saved_basename = start_with_2013(image_00_path)[:-4]
            saved_project_3d_fname_specific_gt = os.path.join(output_folder_gt_only,saved_basename+".png")
            
            
        
            
            
   
            if np.isnan(copied_image_for_gt_labels).any():
                continue
            if np.isnan(copied_image_for_estimate).any():
                continue
            if np.isnan(copied_image_for_gt_and_est).any():
                continue
            
                
            if not os.path.exists(saved_project_3d_fname_specific_gt):
                os.makedirs(os.path.dirname(saved_project_3d_fname_specific_gt),exist_ok=True)
                skimage.io.imsave(saved_project_3d_fname_specific_gt,copied_image_for_gt_labels)
            
            saved_project_3d_fname_specific_est = os.path.join(output_folder_projected_3D_only_path,saved_basename+".png")
            if not os.path.exists(saved_project_3d_fname_specific_est):
                os.makedirs(os.path.dirname(saved_project_3d_fname_specific_est),exist_ok=True)
                skimage.io.imsave(saved_project_3d_fname_specific_est,copied_image_for_estimate)
            
            saved_project_3d_fname_specific_est_and_gt = os.path.join(output_folder_projected_3D_And_GT_path,saved_basename+".png")
            if not os.path.exists(saved_project_3d_fname_specific_est_and_gt):
                os.makedirs(os.path.dirname(saved_project_3d_fname_specific_est_and_gt),exist_ok=True)
                skimage.io.imsave(saved_project_3d_fname_specific_est_and_gt,copied_image_for_gt_and_est)
                        

            # saved the segmentation mask
            segmentation_mask = instance_to_segmentation(instance_mask=gt_mask.cpu().numpy())
            colored_segmentation_mask = instance_to_colored_segmentation(instance_mask=gt_mask.cpu().numpy())


            saved_seg_mask_fname = os.path.join(output_seg_mask_path,saved_basename+".png")
            if not os.path.exists(saved_seg_mask_fname):
                os.makedirs(os.path.dirname(saved_seg_mask_fname),exist_ok=True)
                skimage.io.imsave(saved_seg_mask_fname,(segmentation_mask*255).astype(np.uint8))
    

            saved_ins_mask_fname = os.path.join(output_instance_mask_path,saved_basename+".png")
            if not os.path.exists(saved_ins_mask_fname):
                os.makedirs(os.path.dirname(saved_ins_mask_fname),exist_ok=True)
                skimage.io.imsave(saved_ins_mask_fname,colored_segmentation_mask)


            # saved dynamic mask
            gt_boxes2d_data = np.array([obj.box2d for obj in gt_object_data])
            gt_matched_index_from_mask, gt_matched_index_from_label = bilateral_matching_linear_sum_assignment(boxes_A=gt_2d_bounding_box_from_mask,
                                                                                         boxes_B=gt_boxes2d_data)
            gt_mask = gt_mask[gt_matched_index_from_mask]
            dynamic_list = dynamic_mask_initial[gt_matched_index_from_label]
            

            #FIXME
            dynamic_list = [0,1,1]
            
            dynamic_mask = convert_into_dynamic_mask(instance_mask=gt_mask,
                                      dynamic_mask=dynamic_list)
            saved_dynamic_mask = os.path.join(output_dynamic_mask,saved_basename+".png")
            
            # FIXME
            if not os.path.exists(saved_dynamic_mask):
                os.makedirs(os.path.dirname(saved_dynamic_mask),exist_ok=True)
                skimage.io.imsave(saved_dynamic_mask,(dynamic_mask*255).astype(np.uint8))

            os.makedirs(os.path.dirname(saved_dynamic_mask),exist_ok=True)
            skimage.io.imsave(saved_dynamic_mask,(dynamic_mask*255).astype(np.uint8))
            quit()
                
            
            # confidnce mask
            confidence_mask = confidence_mask_initial[gt_matched_index_from_label]
            
            saved_confidence_mask_fname = os.path.join(output_confidnce_mask,saved_basename+".png")
            if not os.path.exists(saved_confidence_mask_fname):
                os.makedirs(os.path.dirname(saved_confidence_mask_fname),exist_ok=True)
                
                visualize_instance_masks_on_image_and_save(background_image=image_data,masks=gt_mask,
                                                           confidences=confidence_mask,file_path=saved_confidence_mask_fname)
            
            # confidence mask with pd
            saved_confidence_mask_fname_with_pd = os.path.join(output_confidnce_mask_with_pd,saved_basename+".png")

            if not os.path.exists(saved_confidence_mask_fname_with_pd):
                os.makedirs(os.path.dirname(saved_confidence_mask_fname_with_pd),exist_ok=True)
                
                
                
                visualize_instance_masks_on_image_and_save(background_image=copied_image_for_estimate,masks=gt_mask,
                                                           confidences=confidence_mask,file_path=saved_confidence_mask_fname_with_pd)



            

            
            
            


                    
                    
                    

                    
                    

                    
                    
                        
                    

                    
                    
                    
            





                
            
                
                


                                                                                                                                                                                                                                                                                                                                                                                                        



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

