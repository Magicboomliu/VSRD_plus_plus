#!/usr/bin/env python3
"""Rewrite ablation filename lists to repo-relative paths under DATASET.ROOT.

Converts legacy absolute paths, e.g.
  /data/dataset/KITTI/KITTI360_For_Docker/data_2d_raw/...
→
  data_2d_raw/...

Usage:
  python preprocessing/data_organization/normalize_ablation_filenames.py \\
    --root /media/zliu/data12/dataset/KITTI/KITTI360_For_Upload
"""

from __future__ import annotations

import argparse
import os
import re


def _to_rel_image_path(image_path: str, dataset_root: str) -> str:
    path = image_path.strip()
    if not path:
        return path

    if dataset_root and os.path.isabs(path):
        try:
            rel = os.path.relpath(path, dataset_root)
            if not rel.startswith(".."):
                return rel.replace("\\", "/")
        except ValueError:
            pass

    for marker in ("/data_2d_raw/", "data_2d_raw/"):
        idx = path.find(marker)
        if idx >= 0:
            rel = path[idx + 1 :] if marker.startswith("/") else path[idx:]
            return rel.replace("\\", "/")

    raise ValueError(f"Cannot normalize image path: {path}")


def normalize_file(in_path: str, out_path: str, dataset_root: str) -> int:
    with open(in_path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]

    out_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(" ")
        if len(parts) < 3:
            raise ValueError(f"Expected 3 fields, got {len(parts)}: {stripped[:120]}")
        parts[1] = _to_rel_image_path(parts[1], dataset_root)
        out_lines.append(" ".join(parts))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
        if out_lines:
            f.write("\n")
    return len(out_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize ablation filename txt paths.")
    parser.add_argument(
        "--root",
        type=str,
        default=os.environ.get(
            "VSRD_DATASET_ROOT",
            "/media/zliu/data12/dataset/KITTI/KITTI360_For_Upload",
        ),
        help="KITTI360 dataset root (DATASET.ROOT).",
    )
    parser.add_argument(
        "--filenames",
        type=str,
        default="filenames/ablations/train_ablation_filenames.txt",
        help="Relative path to train_ablation_filenames.txt under --root.",
    )
    parser.add_argument(
        "--dynamic",
        type=str,
        default="filenames/ablations/train_ablation_dynamic_mask.txt",
        help="Relative path to train_ablation_dynamic_mask.txt under --root.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite inputs (default: write *.normalized.txt then replace).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = os.path.abspath(args.root)
    filenames_path = (
        args.filenames if os.path.isabs(args.filenames) else os.path.join(root, args.filenames)
    )
    dynamic_path = (
        args.dynamic if os.path.isabs(args.dynamic) else os.path.join(root, args.dynamic)
    )

    for path in (filenames_path, dynamic_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    def _write_pair(src: str, dst: str) -> int:
        n = normalize_file(src, dst, root)
        print(f"[normalize] {n} lines -> {dst}")
        return n

    if args.in_place:
        for path in (filenames_path, dynamic_path):
            tmp = f"{path}.tmp"
            _write_pair(path, tmp)
            os.replace(tmp, path)
    else:
        for path in (filenames_path, dynamic_path):
            tmp = f"{path}.normalized"
            _write_pair(path, tmp)
            os.replace(tmp, path)

    sample = open(filenames_path, encoding="utf-8").readline().strip()
    print(f"[sample] {sample[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
