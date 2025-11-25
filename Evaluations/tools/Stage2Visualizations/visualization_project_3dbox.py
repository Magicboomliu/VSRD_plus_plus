import os
import numpy as np
from kitti_utils import get_calib_from_file, get_objects_from_label
from PIL import Image
import matplotlib.pyplot as plt
from draw_projected_3d import draw3d_bbox_2d_projection
from kitti_box_computation import project_to_image

import pycocotools.mask
import torch
import json
import skimage.io
import cv2
from geo_op import rotation_matrix_x,rotation_matrix_y,rotation_matrix_z
import argparse


def read_annotation(annotation_filename,class_names=['car']):

    with open(annotation_filename) as file:
        annotation = json.load(file)

    intrinsic_matrix = torch.as_tensor(annotation["intrinsic_matrix"])
    extrinsic_matrix = torch.as_tensor(annotation["extrinsic_matrix"])

    # Extract masks for cars
    instance_ids = {
        class_name: list(masks.keys())
        for class_name, masks in annotation["masks"].items()
        if class_name in class_names
    }

    if instance_ids:
        masks = torch.cat([
            torch.as_tensor([
                pycocotools.mask.decode(annotation["masks"][class_name][instance_id])
                for instance_id in instance_ids
            ], dtype=torch.float)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        return dict(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            masks=masks,
            instance_ids=instance_ids,
        )
    else:
        return dict(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            masks=None,
            instance_ids=None,
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


def main(args):
    
    root_folder = args.root_folder
    pd_folder = args.pd_folder
    stage2_vis_output_folder = os.path.join(args.stage2_vis_output_folder,args.options)
    
    os.makedirs(stage2_vis_output_folder,exist_ok=True)
    sync_file = os.path.join(root_folder,'training/sync_file.txt')
    assert os.path.exists(sync_file)
    
    sync_context = read_text_lines(sync_file)

    
    source_image_filename_left_list = []
    saved_image_filename_left_list = []
    source_label_est_name_list = []
    saved_label_est_name_list = []
    source_label_2_name_list = []
    saved_label_2_name_list = []
    
    
    for idx, line in enumerate(sync_context):
        source_image_filename_left,saved_image_left,label_est_name_abs,saved_label_est,label_2_name_abs,saved_label_2 = line.split(" ")
        
        source_image_filename_left_list.append(source_image_filename_left)
        saved_image_filename_left_list.append(saved_image_left)
        source_label_est_name_list.append(label_est_name_abs)
        saved_label_est_name_list.append(saved_label_est)
        source_label_2_name_list.append(label_2_name_abs)
        saved_label_2_name_list.append(saved_label_2)
    

    saved_label_2_name_list_basename = [os.path.basename(path) for path in saved_label_2_name_list]

        
    for idx, pd_label_name in enumerate(sorted(os.listdir(pd_folder))):
        
        pd_label_name_abs_path = os.path.join(pd_folder,pd_label_name)
        gt_label_name_abs_path = os.path.join(root_folder,'training','label_2',pd_label_name)
        left_image_abs_path = gt_label_name_abs_path.replace("label_2","image_2").replace(".txt",'.png')
        calib_abs_path = gt_label_name_abs_path.replace("label_2","calib")
        
        
        
        assert os.path.exists(calib_abs_path)
        assert os.path.exists(left_image_abs_path)
        assert os.path.exists(gt_label_name_abs_path)
        
        
        target_basename = os.path.basename(pd_label_name_abs_path)
        target_index = saved_label_2_name_list_basename.index(target_basename)
        
        target_align_source_txt = source_label_2_name_list[target_index]
        aligned_annotation_folder_path = os.path.join(args.annotation_folder,extract_from_2013(target_align_source_txt)).replace(".txt",'.json')
        assert os.path.exists(aligned_annotation_folder_path)


        annotions_raw_data = read_annotation(aligned_annotation_folder_path,class_names=['car'])
        intrinsic_matrix = torch.as_tensor(annotions_raw_data["intrinsic_matrix"])
        extrinsic_matrix = torch.as_tensor(annotions_raw_data["extrinsic_matrix"])
        target_extrinsic_matrix = extrinsic_matrix
        inverse_target_extrinsic_matrix = torch.linalg.inv(target_extrinsic_matrix)
        
        # Get masks for visualization
        gt_masks = annotions_raw_data.get("masks", None)


        x_axis, y_axis, _ = target_extrinsic_matrix[..., :3, :3]
        rectification_angle = (
            torch.acos(torch.dot(torch.round(y_axis), y_axis)) *
            torch.sign(torch.dot(torch.cross(torch.round(y_axis), y_axis), x_axis))
        )
        rectification_matrix = rotation_matrix_x(rectification_angle).cpu().numpy()


        # image data path
        image_data = read_image(left_image_abs_path)
        # gt 3D bounding boxes
        gt_object_data = get_objects_from_label(gt_label_name_abs_path)
        # estimated 3D bounding boxes
        est_obj_data = get_objects_from_label(pd_label_name_abs_path)
        calib_data = get_calib_from_file(calib_abs_path)
        
        
        # get the object bounding boxes
        copied_image = image_data.copy()
        
        # Overlay segmentation masks on the image
        if gt_masks is not None and len(gt_masks) > 0:
            # Convert masks to numpy and overlay on image
            mask_overlay = copied_image.copy().astype(np.float32)
            image_h, image_w = image_data.shape[:2]
            
            for mask_idx, mask in enumerate(gt_masks):
                mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                # Ensure mask matches image dimensions
                if mask_np.shape[:2] != (image_h, image_w):
                    # Resize mask if needed
                    mask_np = cv2.resize(mask_np.astype(np.float32), (image_w, image_h), interpolation=cv2.INTER_NEAREST)
                
                # Create colored mask overlay (red with transparency for cars)
                mask_color = np.array([255, 0, 0])  # Red color for car masks
                alpha = 0.3  # Transparency
                
                # Expand mask to 3 channels if needed
                if len(mask_np.shape) == 2:
                    mask_3d = np.stack([mask_np, mask_np, mask_np], axis=-1)
                else:
                    mask_3d = mask_np
                
                # Overlay mask on image
                mask_overlay = mask_overlay * (1 - alpha * mask_3d) + mask_color * (alpha * mask_3d)
            
            copied_image = np.clip(mask_overlay, 0, 255).astype(np.uint8)
        
        # Save mask-only image if masks exist
        if gt_masks is not None and len(gt_masks) > 0:
            mask_output_folder = os.path.join(stage2_vis_output_folder, "masks")
            os.makedirs(mask_output_folder, exist_ok=True)
            # Combine all masks into one
            image_h, image_w = image_data.shape[:2]
            combined_mask = np.zeros((image_h, image_w), dtype=np.uint8)
            
            for mask_idx, mask in enumerate(gt_masks):
                mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                # Ensure mask matches image dimensions
                if mask_np.shape[:2] != (image_h, image_w):
                    import cv2
                    mask_np = cv2.resize(mask_np.astype(np.float32), (image_w, image_h), interpolation=cv2.INTER_NEAREST)
                
                # Add mask to combined mask (each instance gets a different value)
                mask_binary = (mask_np > 0.5).astype(np.uint8)
                combined_mask = np.maximum(combined_mask, mask_binary * 255)
            
            # Save as grayscale mask image
            mask_filename = os.path.join(mask_output_folder, target_basename.replace(".txt", "_mask.png"))
            skimage.io.imsave(mask_filename, combined_mask)
        
        for inner_idx, object in enumerate(est_obj_data):

            if args.options =="pd_gt":
                gt_corners3d = gt_object_data[inner_idx].generate_corners3d_raw(rectification_matrix)
                # project gt
                gt_corners2d = project_to_image(gt_corners3d,calib_data["P2"])

                if gt_corners2d.min()<-10:
                    continue
                copied_image = draw3d_bbox_2d_projection(copied_image,qs=gt_corners2d,color=(0,255,0),is_gt=True)
            
            
            # project estimated
            est_corners3d = est_obj_data[inner_idx].generate_corners3d_raw(rectification_matrix)
            weakm3d_corners3d = project_to_image(est_corners3d,calib_data["P2"])
            if weakm3d_corners3d.min()<-0:
                continue
            copied_image = draw3d_bbox_2d_projection(copied_image,qs=weakm3d_corners3d,color=(0,255,0),is_gt=False)
        
        


        skimage.io.imsave(os.path.join(stage2_vis_output_folder,target_basename.replace(".txt",".png")),
                                copied_image)

        if idx%10==0:
            print("Finished {}/{}".format(idx,len(os.listdir(pd_folder))))

        
        
        





if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Stage2 Visualizations Projected 3D Boxes")
    parser.add_argument("--root_folder",type=str,default="datasets/KITTI-360")
    parser.add_argument("--pd_folder",type=str,default="datasets/KITTI-360")
    parser.add_argument("--annotation_folder",type=str,default=None)
    parser.add_argument("--options",type=str,default=None)
    parser.add_argument("--stage2_vis_output_folder",type=str, default="datasets/KITTI-360")
    args = parser.parse_args()

    
    main(args=args)

    
    
    

