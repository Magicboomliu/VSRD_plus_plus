"""Assign per-instance dynamic flags to GT KITTI labels from ``dynamic_mask.txt``."""

from __future__ import annotations

import argparse
import functools
import glob
import json
import logging
import multiprocessing
import os
import sys

import numpy as np
import pycocotools.mask
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

_VALIDATOR_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_VALIDATOR_TOOLS, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from validator.tools.dynamic_labels_io import (
    annotation_to_image_path,
    load_dynamic_mask_by_image,
    lookup_dynamic_labels,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def encode_box_3d(boxes_3d):
    locations = torch.mean(boxes_3d, dim=-2)

    widths = torch.mean(torch.norm(torch.sub(
        boxes_3d[..., [1, 2, 6, 5], :],
        boxes_3d[..., [0, 3, 7, 4], :],
    ), dim=-1), dim=-1)

    heights = torch.mean(torch.norm(torch.sub(
        boxes_3d[..., [4, 5, 6, 7], :],
        boxes_3d[..., [0, 1, 2, 3], :],
    ), dim=-1), dim=-1)

    lengths = torch.mean(torch.norm(torch.sub(
        boxes_3d[..., [1, 0, 4, 5], :],
        boxes_3d[..., [2, 3, 7, 6], :],
    ), dim=-1), dim=-1)

    dimensions = torch.stack([widths, heights, lengths], dim=-1)

    orientations = torch.mean(torch.sub(
        boxes_3d[..., [1, 0, 4, 5], :],
        boxes_3d[..., [2, 3, 7, 6], :],
    ), dim=-2)

    orientations = nn.functional.normalize(orientations[..., [2, 0]], dim=-1)
    orientations = torch.atan2(*reversed(torch.unbind(orientations, dim=-1)))

    return locations, dimensions, orientations


def get_organized_data(annotation_filename, class_names=("car",)):
    with open(annotation_filename, encoding="utf-8") as file:
        annotation = json.load(file)

    if "masks" not in annotation or "boxes_3d" not in annotation:
        return None, None, None, None

    if "car" not in annotation["boxes_3d"] or len(annotation["boxes_3d"]["car"]) == 0:
        return None, None, None, None

    instance_ids = {
        class_name: list(masks.keys())
        for class_name, masks in annotation["masks"].items()
        if class_name in class_names
    }

    gt_boxes_3d = torch.cat([
        torch.as_tensor([
            annotation["boxes_3d"][class_name].get(instance_id, [[np.nan] * 3] * 8)
            for instance_id in ids
        ], dtype=torch.float)
        for class_name, ids in instance_ids.items()
    ], dim=0)

    gt_masks = torch.cat([
        torch.as_tensor(np.stack([
            pycocotools.mask.decode(annotation["masks"][class_name][instance_id])
            for instance_id in ids
        ]), dtype=torch.float)
        for class_name, ids in instance_ids.items()
    ], dim=0)

    gt_boxes_2d = torchvision.ops.masks_to_boxes(gt_masks).unflatten(-1, (2, 2))
    car_instance_ids = instance_ids["car"]
    return car_instance_ids, gt_boxes_3d, gt_boxes_2d


def save_prediction(filename, class_names, boxes_3d, boxes_2d, scores, labels):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as file:
        for class_name, box_3d, box_2d, score, label in zip(
            class_names, boxes_3d, boxes_2d, scores, labels,
        ):
            location, dimension, orientation = encode_box_3d(box_3d)

            location[..., 1] += dimension[..., 1] / 2.0
            dimension = dimension[..., [1, 0, 2]]
            ray_orientation = torch.atan2(
                *reversed(torch.unbind(location[..., [2, 0]], dim=-1)),
            )
            global_orientation = orientation - np.pi / 2.0
            local_orientation = global_orientation - ray_orientation

            file.write(
                f"{class_name.capitalize()} "
                f"{0.0} "
                f"{0} "
                f"{local_orientation} "
                f"{' '.join(map(str, box_2d.flatten().tolist()))} "
                f"{' '.join(map(str, dimension.tolist()))} "
                f"{' '.join(map(str, location.tolist()))} "
                f"{global_orientation} "
                f"{score} "
                f"{label}\n"
            )


def dynamic_attribute_func(
    sequence,
    root_dirname,
    ckpt_dirname,
    class_names,
    json_folder,
    output_labelname,
    dynamic_dirname,
):
    dynamic_by_image = load_dynamic_mask_by_image(dynamic_dirname, sequence, root_dirname)
    prediction_dirname = os.path.join(json_folder, os.path.basename(ckpt_dirname))
    prediction_filenames = sorted(glob.glob(
        os.path.join(root_dirname, prediction_dirname, sequence, "image_00", "data_rect", "*.json"),
    ))
    ckpt_basename = os.path.basename(ckpt_dirname)
    skipped_no_mask = 0
    written = 0

    for prediction_filename in tqdm(prediction_filenames, desc=f"dynamic GT {sequence}"):
        rel_path = os.path.relpath(prediction_filename, root_dirname)
        if rel_path.startswith(f"predictions/{ckpt_basename}/"):
            annotation_rel_path = rel_path.replace(
                f"predictions/{ckpt_basename}/", "annotations/", 1,
            )
        elif rel_path.startswith("predictions/"):
            annotation_rel_path = rel_path.replace("predictions/", "annotations/", 1)
        else:
            annotation_rel_path = rel_path.replace(prediction_dirname, "annotations")
        annotation_filename = os.path.join(root_dirname, annotation_rel_path)

        instance_ids, gt_boxes_3d, gt_boxes_2d = get_organized_data(
            annotation_filename, class_names=class_names,
        )
        if gt_boxes_3d is None:
            continue

        image_path = annotation_to_image_path(root_dirname, annotation_filename)
        dynamic_label_list = lookup_dynamic_labels(dynamic_by_image, image_path, instance_ids)
        if dynamic_label_list is None:
            skipped_no_mask += 1
            continue

        rel_path_no_ext = os.path.splitext(rel_path)[0]
        if rel_path_no_ext.startswith(f"predictions/{ckpt_basename}/"):
            rel_path_no_ext = rel_path_no_ext.replace(
                f"predictions/{ckpt_basename}/",
                f"{output_labelname}/{ckpt_basename}/",
                1,
            )
        elif rel_path_no_ext.startswith("predictions/"):
            rel_path_no_ext = rel_path_no_ext.replace(
                "predictions/", f"{output_labelname}/", 1,
            )
        label_filename = os.path.join(root_dirname, f"{rel_path_no_ext}.txt")

        gt_class_names = ["car"] * len(dynamic_label_list)
        save_prediction(
            filename=label_filename,
            class_names=gt_class_names,
            boxes_3d=gt_boxes_3d,
            boxes_2d=gt_boxes_2d,
            scores=torch.ones(len(gt_class_names)),
            labels=dynamic_label_list,
        )
        written += 1

    logger.info(
        "%s: wrote %d labels, skipped %d frames without dynamic_mask entry",
        sequence, written, skipped_no_mask,
    )


def main(args):
    sequences = list(map(
        os.path.basename,
        sorted(glob.glob(os.path.join(args.root_dirname, "data_2d_raw", "*"))),
    ))

    with multiprocessing.Pool(args.num_workers) as pool:
        with tqdm(total=len(sequences)) as progress_bar:
            for _ in pool.imap_unordered(functools.partial(
                dynamic_attribute_func,
                root_dirname=args.root_dirname,
                ckpt_dirname=args.ckpt_dirname,
                class_names=args.class_names,
                json_folder=args.json_foldername,
                output_labelname=args.output_labelname,
                dynamic_dirname=args.dyanmic_root_filename,
            ), sequences):
                progress_bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign dynamic flags to GT KITTI labels from dynamic_mask.txt",
    )
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-360")
    parser.add_argument("--ckpt_dirname", type=str, default="ckpts/kitti_360/vsrd_plus_plus")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--json_foldername", type=str, default="predictions")
    parser.add_argument("--output_labelname", type=str, default="GT_with_dynamic")
    parser.add_argument(
        "--dyanmic_root_filename",
        type=str,
        required=True,
        help="Directory containing syncXX/dynamic_mask.txt (e.g. dynamic_attributes_est_gt/)",
    )
    main(parser.parse_args())
