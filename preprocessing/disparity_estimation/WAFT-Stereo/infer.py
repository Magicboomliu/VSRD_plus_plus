"""CLI wrapper around :class:`pipeline.DepthEstimationPipeline`."""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import imageio.v2 as imageio
import numpy as np

from pipeline import DepthEstimationPipeline
from visualize import get_heatmap, vis_heatmap


def collect_pairs(data_dir: str):
    left_dir = os.path.join(data_dir, "left")
    right_dir = os.path.join(data_dir, "right")

    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG")
    left_files = sorted(p for ext in exts for p in glob.glob(os.path.join(left_dir, ext)))
    if not left_files:
        raise FileNotFoundError(f"No images found in {left_dir}")

    pairs = []
    for left_path in left_files:
        stem = os.path.splitext(os.path.basename(left_path))[0]
        ext = os.path.splitext(left_path)[1]
        right_path = os.path.join(right_dir, stem + ext)
        if not os.path.exists(right_path):
            print(f"  [skip] no matching right image for {left_path}")
            continue
        pairs.append((stem, left_path, right_path))
    return pairs


def save_outputs(
    out_dir: str,
    left_img: np.ndarray,
    depth: np.ndarray,
    disparity: np.ndarray,
    depth_max: float,
    uncertainty: np.ndarray | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "depth_pred.npy"), depth)
    np.save(os.path.join(out_dir, "disp_pred.npy"), disparity)

    depth_norm = (depth / depth_max * 255).astype(np.uint8)
    vis_depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    vis_depth_rgb = cv2.cvtColor(vis_depth_colour, cv2.COLOR_BGR2RGB)
    imageio.imwrite(
        os.path.join(out_dir, "vis_depth.png"),
        np.concatenate([left_img, vis_depth_rgb], axis=1),
    )

    if uncertainty is not None:
        cv2.imwrite(os.path.join(out_dir, "uncertainty.png"), vis_heatmap(left_img, uncertainty))

    valid = depth > 0
    if valid.any():
        print(
            f"  depth range (valid): [{depth[valid].min():.2f}, {depth[valid].max():.2f}] m  →  {out_dir}"
        )
    else:
        print(f"  no valid depth pixels  →  {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="WAFT-Stereo batch inference")
    parser.add_argument("--data-dir", default="Data_Examples")
    parser.add_argument("--config-file", default="configs/Real/kitti.yaml")
    parser.add_argument("--ckpt", default="ckpts/Real/DAv2L-5.pth")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fx", type=float, default=552.554261)
    parser.add_argument("--baseline", type=float, default=0.5942)
    parser.add_argument("--depth-min", type=float, default=0.0)
    parser.add_argument("--depth-max", type=float, default=80.0)
    args = parser.parse_args()

    out_root = args.out_dir or os.path.join(args.data_dir, "output")
    pairs = collect_pairs(args.data_dir)
    print(f"Found {len(pairs)} stereo pair(s) in {args.data_dir}")
    print(
        f"Camera: fx={args.fx}, baseline={args.baseline} m, "
        f"depth=[{args.depth_min}, {args.depth_max}] m"
    )

    pipeline = DepthEstimationPipeline(
        config_file=args.config_file,
        ckpt_path=args.ckpt,
        fx=args.fx,
        baseline=args.baseline,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )

    for stem, left_path, right_path in pairs:
        print(f"Processing {stem} ...")
        left_img = imageio.imread(left_path)
        depth, disparity = pipeline.infer_with_disparity(left_path, right_path)
        save_outputs(
            os.path.join(out_root, stem),
            left_img=left_img,
            depth=depth,
            disparity=disparity,
            depth_max=args.depth_max,
        )

    print("Done.")


if __name__ == "__main__":
    main()
