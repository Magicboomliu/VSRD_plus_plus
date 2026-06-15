#!/usr/bin/env python3
"""Compare GT-bbox dynamic mask (current pipeline) vs dynamic_mask.txt labels."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.Dynamic_Labels import build_multi_from_annotations, load_sample_lines
from preprocessing.Initial_Attributes.gt_attributes import (
    DEFAULT_DYNAMIC_VELOCITY_THRESHOLD,
    compute_gt_velocity,
    infer_dynamic_mask_from_gt_velocity,
)
from trainer.configs import load_config
from vsrd_plus_plus.datasets.kitti360_dataset import KITTI360Dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sequence_00", help="Trainer config name")
    parser.add_argument("--threshold", type=float, default=DEFAULT_DYNAMIC_VELOCITY_THRESHOLD)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all frames")
    parser.add_argument("--dynamic-path", type=str, default="",
                        help="Override dynamic_mask.txt path (default: config DYNAMIC_LABELS_PATH). "
                             "Compare vs legacy: .../dynamic_attributes_est/syncXX/dynamic_mask.txt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg.TRAIN.DATASET.ROOT
    filenames_path = cfg.TRAIN.DATASET.FILENAMES[0]
    if args.dynamic_path:
        dynamic_path = args.dynamic_path
        if not os.path.isabs(dynamic_path):
            dynamic_path = os.path.join(root, dynamic_path)
    else:
        dynamic_path = cfg.TRAIN.DYNAMIC_LABELS_PATH
    num_source_frames = cfg.TRAIN.DATASET.NUM_SOURCE_FRAMES

    ds = KITTI360Dataset(
        filenames=cfg.TRAIN.DATASET.FILENAMES,
        class_names=cfg.TRAIN.DATASET.CLASS_NAMES,
        dataset_root=root,
    )

    tp = fp = tn = fn = 0
    skipped = 0
    frame_agree = frame_total = 0
    speed_when_legacy_dyn: list[float] = []
    speed_when_legacy_static: list[float] = []
    disagree_examples: list[str] = []

    samples = list(load_sample_lines(filenames_path, dynamic_path, root))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    for idx, (target_img, src_rels, legacy_ids, legacy_labels) in enumerate(samples):
        multi = build_multi_from_annotations(ds, target_img, src_rels, num_source_frames)
        if multi is None:
            skipped += 1
            continue

        gt_v = compute_gt_velocity(multi)
        speeds = gt_v.squeeze(0).norm(dim=-1).tolist()
        predicted = infer_dynamic_mask_from_gt_velocity(gt_v, threshold=args.threshold)

        target_instance_ids = multi[0]["instance_ids"][0].tolist()
        legacy_map = dict(zip(legacy_ids, legacy_labels))

        frame_match = True
        for i, iid in enumerate(target_instance_ids):
            if iid not in legacy_map:
                continue
            gt_label = legacy_map[iid]
            pred = predicted[i]
            spd = speeds[i]

            if gt_label:
                speed_when_legacy_dyn.append(spd)
            else:
                speed_when_legacy_static.append(spd)

            if pred and gt_label:
                tp += 1
            elif pred and not gt_label:
                fp += 1
                frame_match = False
                if len(disagree_examples) < 20:
                    disagree_examples.append(
                        f"FP {os.path.basename(target_img)} id={iid} speed={spd:.6f} m/frame pred=1 legacy=0"
                    )
            elif not pred and gt_label:
                fn += 1
                frame_match = False
                if len(disagree_examples) < 20:
                    disagree_examples.append(
                        f"FN {os.path.basename(target_img)} id={iid} speed={spd:.6f} m/frame pred=0 legacy=1"
                    )
            else:
                tn += 1

        frame_total += 1
        if frame_match:
            frame_agree += 1

        if (idx + 1) % 500 == 0:
            print(f"  processed {idx + 1}/{len(samples)} frames...", flush=True)

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Reference: {dynamic_path}")
    print(f"Threshold: {args.threshold} m/frame")
    print(f"Frames: {frame_total} (skipped {skipped})")
    print("=" * 60)
    print("Confusion matrix (pred rows vs reference cols):")
    print(f"              ref=0    ref=1")
    print(f"  pred=0        {tn:5d}       {fn:5d}")
    print(f"  pred=1        {fp:5d}       {tp:5d}")
    print()
    print(f"Instance accuracy:  {accuracy*100:.2f}%  ({tp+tn}/{total})")
    print(f"Frame-exact match:  {frame_agree}/{frame_total} ({100*frame_agree/frame_total:.2f}%)")
    print(f"Dynamic precision:  {precision*100:.2f}%")
    print(f"Dynamic recall:     {recall*100:.2f}%")
    print(f"Dynamic F1:         {f1*100:.2f}%")
    print(f"Reference dynamic rate: {(tp+fn)/total*100:.2f}%  ({tp+fn}/{total})")
    print(f"Pred dynamic rate:   {(tp+fp)/total*100:.2f}%  ({tp+fp}/{total})")

    if speed_when_legacy_static:
        arr = np.array(speed_when_legacy_static)
        print(f"\nSpeed m/frame when ref=0: med={np.median(arr):.6f} p95={np.percentile(arr,95):.6f} max={arr.max():.4f}")
    if speed_when_legacy_dyn:
        arr = np.array(speed_when_legacy_dyn)
        print(f"Speed m/frame when ref=1: med={np.median(arr):.6f} p95={np.percentile(arr,95):.6f} max={arr.max():.4f}")

    print("\nSample disagreements:")
    for line in disagree_examples[:15]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
