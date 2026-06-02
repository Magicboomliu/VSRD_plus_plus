"""Build aligned multi-frame dicts from KITTI360 annotations (no images / depth)."""

from __future__ import annotations

import numpy as np
import torch

from vsrd_plus_plus.datasets.kitti360_dataset import KITTI360Dataset


def build_multi_from_annotations(
    ds: KITTI360Dataset,
    target_img: str,
    source_relative_indices: list[int],
    num_source_frames: int,
) -> dict | None:
    """Return ``multi_inputs`` keyed by relative frame index, or ``None`` if too sparse."""
    split = np.array_split(source_relative_indices, num_source_frames)
    middle_indices = [int(s[len(s) // 2]) for s in split if s.size > 0]
    frame_keys = sorted(set([0] + middle_indices))

    raw: dict[int, dict] = {}
    for rel in frame_keys:
        path = KITTI360Dataset.get_image_filename(target_img, rel)
        ann = ds.read_annotation(KITTI360Dataset.get_annotation_filename(path))
        if "instance_ids" in ann:
            raw[rel] = ann
    if len(raw) < 2:
        return None

    multi: dict[int, dict] = {}
    for rel, ann in raw.items():
        multi[rel] = {
            "extrinsic_matrices": ann["extrinsic_matrix"].unsqueeze(0),
            "boxes_3d": ann["boxes_3d"].unsqueeze(0),
            "instance_ids": ann["instance_ids"].unsqueeze(0),
        }

    target_ids = multi[0]["instance_ids"][0]
    num_instances = target_ids.shape[0]
    for frame in multi.values():
        src_ids = frame["instance_ids"][0]
        aligned = torch.tensor(
            [src_ids.tolist().index(t.item()) if t in src_ids else -1 for t in target_ids]
        )
        boxes = torch.full((1, num_instances, 8, 3), float("nan"))
        visible = aligned >= 0
        if visible.any():
            boxes[0, visible] = frame["boxes_3d"][0, aligned[visible]]
        frame["boxes_3d"] = boxes
        frame["instance_ids"] = multi[0]["instance_ids"].clone()
        frame["visible_masks"] = [visible]
    multi[0]["visible_masks"] = [torch.ones(num_instances, dtype=torch.bool)]
    return multi
