#!/usr/bin/env python3
"""Sweep dynamic velocity threshold across all sequence configs vs dynamic_mask.txt."""

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
)
from trainer.configs import load_config
from vsrd_plus_plus.datasets.kitti360_dataset import KITTI360Dataset

ALL_SEQUENCES = [
    "sequence_00", "sequence_02", "sequence_03", "sequence_04",
    "sequence_05", "sequence_06", "sequence_07", "sequence_09", "sequence_10",
]


def cache_sequence(config_name: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    cfg = load_config(config_name)
    ds = KITTI360Dataset(
        filenames=cfg.TRAIN.DATASET.FILENAMES,
        class_names=cfg.TRAIN.DATASET.CLASS_NAMES,
        dataset_root=cfg.TRAIN.DATASET.ROOT,
    )
    samples = list(load_sample_lines(
        cfg.TRAIN.DATASET.FILENAMES[0],
        cfg.TRAIN.DYNAMIC_LABELS_PATH,
        cfg.TRAIN.DATASET.ROOT,
    ))
    speeds, labels = [], []
    skipped = 0
    for idx, (target_img, src_rels, legacy_ids, legacy_labels) in enumerate(samples):
        multi = build_multi_from_annotations(
            ds, target_img, src_rels, cfg.TRAIN.DATASET.NUM_SOURCE_FRAMES,
        )
        if multi is None:
            skipped += 1
            continue
        gt_v = compute_gt_velocity(multi)
        spd = gt_v.squeeze(0).norm(dim=-1).tolist()
        legacy_map = dict(zip(legacy_ids, legacy_labels))
        for i, iid in enumerate(multi[0]["instance_ids"][0].tolist()):
            if iid in legacy_map:
                speeds.append(spd[i])
                labels.append(legacy_map[iid])
        if (idx + 1) % 500 == 0:
            print(f"  {config_name}: {idx + 1}/{len(samples)} frames", flush=True)
    return np.array(speeds), np.array(labels, dtype=bool), len(samples), skipped


def metrics(speeds: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    pred = speeds >= thr
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    tn = int((~pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, acc=acc, prec=prec, rec=rec, f1=f1, total=total)


def best_threshold(
    speeds: np.ndarray,
    labels: np.ndarray,
    *,
    min_recall: float = 1.0,
    optimize: str = "acc",
) -> tuple[float, dict]:
    grid = [0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03,
            0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15,
            0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.50]
    if labels.any():
        grid.extend(float(x) for x in speeds[labels])
    candidates = sorted(set(grid))

    best_thr, best_m = candidates[0], None
    for thr in candidates:
        m = metrics(speeds, labels, thr)
        if m["rec"] + 1e-9 < min_recall:
            continue
        score = m[optimize]
        if best_m is None or score > best_m[optimize] or (
            score == best_m[optimize] and thr < best_thr
        ):
            best_thr, best_m = thr, m
    assert best_m is not None
    return best_thr, best_m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences", nargs="*", default=ALL_SEQUENCES,
        help=f"Config names (default: all {len(ALL_SEQUENCES)} sequences)",
    )
    args = parser.parse_args()

    per_seq: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in args.sequences:
        print(f"\nCaching {name}...", flush=True)
        speeds, labels, n_frames, skipped = cache_sequence(name)
        per_seq[name] = (speeds, labels)
        dyn = labels.sum()
        print(
            f"  {name}: {n_frames} frames ({skipped} skipped), "
            f"{len(speeds)} instances, legacy dynamic={dyn} ({100*dyn/max(len(labels),1):.1f}%)",
            flush=True,
        )
        if len(speeds):
            s0 = speeds[~labels] if (~labels).any() else np.array([0.0])
            s1 = speeds[labels] if labels.any() else np.array([0.0])
            print(
                f"  speed static max={s0.max():.4f}  dynamic min={s1.min():.4f}  "
                f"dynamic med={np.median(s1):.4f}",
                flush=True,
            )

    all_speeds = np.concatenate([s for s, _ in per_seq.values()])
    all_labels = np.concatenate([l for _, l in per_seq.values()])

    print("\n" + "=" * 72)
    print(f"GLOBAL: {len(all_speeds)} instances, legacy dynamic={all_labels.sum()} "
          f"({100*all_labels.mean():.2f}%)")
    print("=" * 72)

    m_default = metrics(all_speeds, all_labels, DEFAULT_DYNAMIC_VELOCITY_THRESHOLD)
    print(f"\nCurrent default {DEFAULT_DYNAMIC_VELOCITY_THRESHOLD} m/frame:")
    print(f"  acc={100*m_default['acc']:.2f}%  F1={100*m_default['f1']:.2f}%  "
          f"FP={m_default['fp']} FN={m_default['fn']}")

    for objective, min_rec, title in [
        ("acc", 1.0, "Best accuracy (recall=100%)"),
        ("f1", 1.0, "Best F1 (recall=100%)"),
        ("acc", 0.99, "Best accuracy (recall>=99%)"),
    ]:
        thr, m = best_threshold(all_speeds, all_labels, min_recall=min_rec, optimize=objective)
        print(f"\n{title}:")
        print(f"  threshold={thr:.4f} m/frame")
        print(f"  acc={100*m['acc']:.2f}%  prec={100*m['prec']:.2f}%  rec={100*m['rec']:.2f}%  "
              f"F1={100*m['f1']:.2f}%  FP={m['fp']} FN={m['fn']}")

    print("\n--- Threshold grid (global) ---")
    print(f"{'thr':>8} {'acc%':>7} {'prec%':>7} {'rec%':>7} {'F1%':>7} {'FP':>6} {'FN':>5}")
    show_thrs = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.18, 0.20, 0.21, 0.22, 0.25, 0.30]
    best_acc_r100, _ = best_threshold(all_speeds, all_labels, min_recall=1.0, optimize="acc")
    show_thrs.append(best_acc_r100)
    for thr in sorted(set(show_thrs)):
        m = metrics(all_speeds, all_labels, thr)
        mark = " <-- best@rec100" if abs(thr - best_acc_r100) < 1e-6 else ""
        print(f"{thr:8.4f} {100*m['acc']:7.2f} {100*m['prec']:7.2f} {100*m['rec']:7.2f} "
              f"{100*m['f1']:7.2f} {m['fp']:6d} {m['fn']:5d}{mark}")

    print("\n--- Per-sequence @ global best (recall=100%) ---")
    gthr, gm = best_threshold(all_speeds, all_labels, min_recall=1.0, optimize="acc")
    print(f"Using threshold={gthr:.4f} m/frame\n")
    print(f"{'seq':>12} {'inst':>6} {'acc%':>7} {'FP':>5} {'FN':>4}")
    for name in args.sequences:
        s, l = per_seq[name]
        m = metrics(s, l, gthr)
        print(f"{name.replace('sequence_', 'sync'):>12} {m['total']:6d} {100*m['acc']:7.2f} "
              f"{m['fp']:5d} {m['fn']:4d}")


if __name__ == "__main__":
    main()
