#!/usr/bin/env python3
"""
Visualize KITTI360 GT 3D boxes.

Outputs (per sample):
- projected 3D boxes on target image (optionally with GT masks)
- BEV boxes

Example:
  pixi run python trainer/visualize_gt.py --config_path ablation_selective --index 0 --out_dir /tmp/vsrdpp_vis
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch

# Make imports work when executed as a script from any cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from trainer.configs import load_config
from vsrd_plus_plus.datasets.kitti360_dataset import KITTI360Dataset
from vsrd_plus_plus.datasets.transforms import (
    Resizer,
    MaskAreaFilter,
    MaskRefiner,
    BoxGenerator,
    BoxSizeFilter,
    SoftRasterizer,
)
from vsrd_plus_plus import visualization


LINE_INDICES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]


def _to_uint8_chw(img_chw: torch.Tensor) -> torch.Tensor:
    img = img_chw.detach().cpu()
    if img.dtype != torch.uint8:
        img = (img.float().clamp(0, 1) * 255.0).byte()
    return img


def _save_image(path: str, img_chw: torch.Tensor) -> None:
    import skimage.io

    img = img_chw.detach().cpu()
    if img.dtype != torch.uint8:
        img = (img.float().clamp(0, 1) * 255.0).byte()
    img_hwc = img.permute(1, 2, 0).numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    skimage.io.imsave(path, img_hwc)


def make_bev_canvas(height: int = 512, width: int = 512) -> torch.Tensor:
    # white background
    return torch.full((3, height, width), 255, dtype=torch.uint8)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="ablation_selective")
    parser.add_argument("--index", type=int, default=0, help="Dataset sample index (used if --indices not set).")
    parser.add_argument("--indices", type=str, default="", help='Comma-separated indices, e.g. "0,1,2,3".')
    parser.add_argument("--start", type=int, default=0, help="Start index for a range run.")
    parser.add_argument("--count", type=int, default=0, help="How many samples to render starting at --start (0 disables).")
    parser.add_argument("--out_dir", type=str, default="trainer/results_visualizations/gt")
    parser.add_argument("--draw_masks", action="store_true", help="Overlay GT masks on the image.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_path)

    dataset_root = cfg.TRAIN.DATASET.ROOT
    class_names = cfg.TRAIN.DATASET.CLASS_NAMES
    num_of_workers = cfg.TRAIN.DATASET.NUMS_OF_WORKERS
    num_source_frames = cfg.TRAIN.DATASET.NUM_SOURCE_FRAMES
    dataset_rectification = cfg.TRAIN.DATASET.RECTIFICATION

    # Use the same transform stack as training so image size / boxes_2d are consistent.
    tgt_sz = cfg.TRAIN.DATASET.TARGET_TRANSFORMS.IMAGE_SIZE
    tgt_min_area1 = cfg.TRAIN.DATASET.TARGET_TRANSFORMS.MIN_MASK_AREA_01
    tgt_min_area2 = cfg.TRAIN.DATASET.TARGET_TRANSFORMS.MIN_MASK_AREA_02
    tgt_min_box = cfg.TRAIN.DATASET.TARGET_TRANSFORMS.MIN_BOX_SIZE

    target_transforms = [
        Resizer(image_size=tgt_sz),
        MaskAreaFilter(min_mask_area=tgt_min_area1),
        MaskRefiner(),
        MaskAreaFilter(min_mask_area=tgt_min_area2),
        BoxGenerator(),
        BoxSizeFilter(min_box_size=tgt_min_box),
        SoftRasterizer(),
    ]

    # Source transforms aren't needed for GT visualization, but dataset requires them for multi-view packing.
    src_sz = cfg.TRAIN.DATASET.SOURCE_TRANSFORMS.IMAGE_SIZE
    src_min_area1 = cfg.TRAIN.DATASET.SOURCE_TRANSFORMS.MIN_MASK_AREA_01
    src_min_area2 = cfg.TRAIN.DATASET.SOURCE_TRANSFORMS.MIN_MASK_AREA_02
    src_min_box = cfg.TRAIN.DATASET.SOURCE_TRANSFORMS.MIN_BOX_SIZE
    source_transforms = [
        Resizer(image_size=src_sz),
        MaskAreaFilter(min_mask_area=src_min_area1),
        MaskRefiner(),
        MaskAreaFilter(min_mask_area=src_min_area2),
        BoxGenerator(),
        BoxSizeFilter(min_box_size=src_min_box),
        SoftRasterizer(),
    ]

    ds = KITTI360Dataset(
        filenames=cfg.TRAIN.DATASET.FILENAMES,
        class_names=class_names,
        num_of_workers=num_of_workers,
        num_source_frames=num_source_frames,
        target_transforms=target_transforms,
        source_transforms=source_transforms,
        rectification=dataset_rectification,
        dataset_root=dataset_root,
    )

    # Decide which indices to render.
    idxs: list[int] = []
    if args.indices.strip():
        idxs = [int(x) for x in args.indices.split(",") if x.strip() != ""]
    elif args.count and args.count > 0:
        idxs = list(range(args.start, args.start + args.count))
    else:
        idxs = [args.index]

    for idx in idxs:
        multi_inputs = ds[idx]
        target = multi_inputs[0]

        image_filename = target["filename"]
        image_dirname = os.path.splitext(
            os.path.relpath(image_filename, KITTI360Dataset.get_root_dirname(image_filename))
        )[0]

        img = _to_uint8_chw(target["image"])
        intrinsic = target["intrinsic_matrix"]
        gt_boxes_3d = target.get("boxes_3d", torch.empty(0, 8, 3))

        overlay = img
        if args.draw_masks:
            masks = target.get("hard_masks", None)
            if masks is None:
                masks = target.get("masks", None)
            if masks is not None and getattr(masks, "numel", lambda: 0)() > 0:
                overlay = visualization.draw_masks(overlay.float() / 255.0, masks.float(), weight=0.35)
                overlay = _to_uint8_chw(overlay)

        if gt_boxes_3d is not None and len(gt_boxes_3d) > 0:
            overlay = visualization.draw_boxes_3d(
                overlay,
                gt_boxes_3d,
                LINE_INDICES,
                intrinsic,
                color=(255, 0, 0),
                thickness=2,
            )

        out_base = os.path.join(args.out_dir, image_dirname)
        _save_image(os.path.join(out_base, "gt_projected_3d.png"), overlay)

        bev = make_bev_canvas()
        if gt_boxes_3d is not None and len(gt_boxes_3d) > 0:
            bev = visualization.draw_boxes_bev(bev, gt_boxes_3d, color=(255, 0, 0), thickness=2)
        _save_image(os.path.join(out_base, "gt_bev.png"), bev)

        print(f"[visualize_gt] idx={idx} saved to: {out_base}")


if __name__ == "__main__":
    main()

