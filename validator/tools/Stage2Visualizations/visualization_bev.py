import os
import numpy as np

from kitti_utils import get_calib_from_file, get_objects_from_label,get_cars_from_label

from PIL import Image
import matplotlib.pyplot as plt
from draw_projected_3d import draw3d_bbox_2d_projection
from kitti_box_computation import project_to_image

import pycocotools.mask
import torch
import json
import skimage.io
from geo_op import rotation_matrix_x,rotation_matrix_y,rotation_matrix_z
import cv2
import argparse

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
            torch.as_tensor([
                pycocotools.mask.decode(annotation["masks"][class_name][instance_id])
                for instance_id in instance_ids
            ], dtype=torch.float)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        labels = torch.cat([
            torch.as_tensor([class_names.index(class_name)] * len(instance_ids), dtype=torch.long)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

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


def read_image(image_path):
    image_data = np.array(Image.open(image_path))
    return image_data



def main(args):

    root_folder = args.root_folder
    pd_folder = args.pd_folder
    stage2_vis_output_folder = args.stage2_vis_output_folder

    os.makedirs(stage2_vis_output_folder,exist_ok=True)

    for idx, pd_label_name in enumerate(sorted(os.listdir(pd_folder))):
        
        pd_label_name_abs_path = os.path.join(pd_folder,pd_label_name)
        gt_label_name_abs_path = os.path.join(root_folder,'training','label_2',pd_label_name)
        left_image_abs_path = gt_label_name_abs_path.replace("label_2","image_2").replace(".txt",'.png')
        calib_abs_path = gt_label_name_abs_path.replace("label_2","calib")
        
    
        assert os.path.exists(calib_abs_path)
        assert os.path.exists(left_image_abs_path)
        assert os.path.exists(gt_label_name_abs_path)
        
        
        target_basename = os.path.basename(pd_label_name_abs_path)

        image_data = read_image(left_image_abs_path)
        gt_object_data = get_cars_from_label(gt_label_name_abs_path)
        # estimated 3D bounding boxes
        try:
            est_obj_data = get_objects_from_label(pd_label_name_abs_path)
        except:
            continue
    

        length_corners_gt = len(gt_object_data)
        length_corners_est = len(est_obj_data)

        if length_corners_gt>=length_corners_est:
            smaller_nums = length_corners_est
        else:
            smaller_nums = length_corners_gt
        
        if smaller_nums==0:
            continue
        bevs = (np.ones((1000,1000,3))*255).astype(np.uint8)
        for inner_idx in range(smaller_nums):
            gt_corners3d = gt_object_data[inner_idx].generate_corners3d()
            est_corners3d = est_obj_data[inner_idx].generate_corners3d()

            
            bevs = draw_boxes_bev(image=bevs,
                                    color=(255, 0, 0),
                                    thickness=2,
                                    extents=((-50.0, 100.0), (50.0, 0.0)),
                                    lineType=cv2.LINE_AA,
                                    boxes_3d=gt_corners3d)
            bevs = draw_boxes_bev( 
                            image=bevs,
                            color=(0, 0, 255),
                            thickness=2,
                            extents=((-50.0, 100.0), (50.0, 0.0)),
                            lineType=cv2.LINE_AA,
                            boxes_3d=est_corners3d)
        
        bevs= bevs.astype(np.uint8)
        skimage.io.imsave(os.path.join(stage2_vis_output_folder,target_basename.replace(".txt",".png")).format(idx),bevs)
        if idx%10==0:
            print("Finished {}/{}".format(idx,len(os.listdir(pd_folder))))



if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Stage2 Visualizations Projected 3D Boxes")
    parser.add_argument("--root_folder",type=str,default="datasets/KITTI-360")
    parser.add_argument("--pd_folder",type=str,default="datasets/KITTI-360")
    parser.add_argument("--stage2_vis_output_folder",type=str, default="datasets/KITTI-360")
    args = parser.parse_args()
    main(args=args)
    
    