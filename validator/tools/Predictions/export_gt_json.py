#!/usr/bin/env python3
"""Export GT boxes_3d / boxes_2d as JSON (legacy Step1b, split-list mode)."""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
import os
import sys

import numpy as np
import pycocotools.mask
import torch
import torchvision
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from validator.tools.Predictions.make_predictions import prediction_json_path
from validator.tools.split_list_io import FilenameListEntry, load_filename_list_entries
from vsrd_plus_plus.transforms import MaskRefiner


def _get_box_3d(annotation: dict, class_name: str, instance_id: int):
    boxes = annotation["boxes_3d"][class_name]
    key = str(instance_id)
    if key in boxes:
        return boxes[key]
    return boxes.get(instance_id, [[float("nan")] * 3] * 8)


def export_gt_json_for_entry(
    entry: FilenameListEntry,
    root_dirname: str,
    class_names: list[str],
    json_out_dirname: str,
) -> None:
    for source_image_filename in entry.source_image_paths:
        source_annotation_filename = source_image_filename.replace(
            "data_2d_raw", "annotations"
        ).replace(".png", ".json")
        if not os.path.exists(source_annotation_filename):
            print(f"[gt] missing annotation: {source_annotation_filename}")
            continue

        with open(source_annotation_filename, encoding="utf-8") as file:
            source_annotation = json.load(file)

        source_boxes_3d = torch.cat(
            [
                torch.as_tensor(
                    [_get_box_3d(source_annotation, class_name, iid) for iid in entry.instance_ids],
                    dtype=torch.float,
                )
                for class_name in class_names
                if class_name in source_annotation.get("boxes_3d", {})
            ],
            dim=0,
        ).unsqueeze(0)

        source_gt_masks = torch.cat(
            [
                torch.as_tensor(
                    np.stack(list(map(pycocotools.mask.decode, masks.values()))),
                    dtype=torch.float,
                )
                for class_name, masks in source_annotation["masks"].items()
                if class_name in class_names
            ],
            dim=0,
        )
        source_gt_masks = MaskRefiner()(source_gt_masks)["masks"]
        source_gt_boxes_2d = torchvision.ops.masks_to_boxes(
            source_gt_masks.bool()
        ).unflatten(-1, (2, 2))

        out_filename = prediction_json_path(
            source_annotation_filename,
            root_dirname,
            json_out_dirname=json_out_dirname,
        )
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        with open(out_filename, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "boxes_3d": {class_names[0]: source_boxes_3d.squeeze(0).tolist()},
                    "boxes_2d": {class_names[0]: source_gt_boxes_2d.tolist()},
                },
                file,
                indent=4,
                sort_keys=False,
            )


def main_from_split_list(args: argparse.Namespace) -> None:
    entries = load_filename_list_entries(
        args.filenames_list,
        args.dynamic_labels_path,
        args.root_dirname,
    )
    print(f"[gt] split list: {len(entries)} instance groups from {args.filenames_list}")

    worker = functools.partial(
        export_gt_json_for_entry,
        root_dirname=args.root_dirname,
        class_names=args.class_names,
        json_out_dirname=args.json_out_dirname,
    )
    with multiprocessing.Pool(args.num_workers) as pool:
        with tqdm(total=len(entries)) as progress_bar:
            for _ in pool.imap_unordered(worker, entries):
                progress_bar.update(1)


def main(args: argparse.Namespace) -> None:
    if not args.filenames_list:
        raise SystemExit("export_gt_json requires --filenames_list")
    if not args.json_out_dirname:
        raise SystemExit("export_gt_json requires --json_out_dirname")
    main_from_split_list(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GT JSON from split filename list")
    parser.add_argument("--root_dirname", type=str, required=True)
    parser.add_argument("--filenames_list", type=str, required=True)
    parser.add_argument("--dynamic_labels_path", type=str, required=True)
    parser.add_argument("--json_out_dirname", type=str, required=True)
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--num_workers", type=int, default=4)
    main(parser.parse_args())
