"""End-to-end pipeline for initial 3D attribute estimation."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
from tqdm import tqdm

from preprocessing.Initial_Attributes.file_io_utils import (
    get_depth_filename,
    read_depth,
    read_pickle_file,
)
from preprocessing.Initial_Attributes.gt_attributes import (
    DEFAULT_DYNAMIC_VELOCITY_THRESHOLD,
    compute_gt_attributes,
    compute_gt_velocity,
    infer_dynamic_mask_from_gt_velocity,
)
from preprocessing.Initial_Attributes.keys import AttributeKeys
from preprocessing.Initial_Attributes.location_orientation import estimate_location_orientation
from preprocessing.Initial_Attributes.roi_lidar import build_roi_lidar
from preprocessing.Initial_Attributes.velocity import estimate_velocity_icp


@dataclass
class InitialAttributesConfig:
    """Hyper-parameters for the initial-attribute pipeline."""

    min_roi_points: int = 120
    min_points_for_icp: int = 100
    device: str = "cuda:0"
    compute_gt: bool = True
    show_progress: bool = True
    dynamic_velocity_threshold: float = DEFAULT_DYNAMIC_VELOCITY_THRESHOLD


def attach_pseudo_depth(multi_inputs: dict) -> dict:
    """Load pseudo-depth maps and attach them to each frame dict."""
    for frame_inputs in multi_inputs.values():
        image_path = frame_inputs["filenames"][0]
        depth_path = get_depth_filename(image_path)
        assert os.path.exists(depth_path), f"Missing pseudo depth: {depth_path}"

        depth = torch.from_numpy(read_depth(depth_path))
        depth = depth.unsqueeze(0).unsqueeze(0).to(frame_inputs["images"].device)
        frame_inputs[AttributeKeys.PSEUDO_DEPTH] = depth

    return multi_inputs


class InitialAttributesPipeline:
    """RoI LiDAR → GT-based dynamic mask → ICP velocity → location/orientation."""

    def __init__(self, config: InitialAttributesConfig | None = None):
        self.config = config or InitialAttributesConfig()

    def run(self, multi_inputs: dict) -> dict:
        device = self.config.device
        multi_inputs = attach_pseudo_depth(multi_inputs)

        target_instance_ids = (
            multi_inputs[0]["instance_ids"][0].cpu().numpy().tolist()
        )
        frame_keys = sorted(multi_inputs.keys())
        iterator = tqdm(frame_keys) if self.config.show_progress else frame_keys

        # Step 1: pseudo RoI LiDAR per reference frame.
        for frame_key in iterator:
            roi_lidar, projected_mask = build_roi_lidar(
                frame_inputs=multi_inputs[frame_key],
                target_instance_ids=target_instance_ids,
            )
            multi_inputs[frame_key][AttributeKeys.ROI_LIDAR] = roi_lidar
            multi_inputs[frame_key][AttributeKeys.PROJECTED_VALID_MASK] = projected_mask
            multi_inputs[frame_key][AttributeKeys.ROI_LIDAR_VALID] = projected_mask is not None

        # Step 2: infer dynamic / static from GT bbox velocity (no external mask file).
        gt_velocity = compute_gt_velocity(multi_inputs)
        dynamic_mask_list = infer_dynamic_mask_from_gt_velocity(
            gt_velocity,
            threshold=self.config.dynamic_velocity_threshold,
        )
        multi_inputs[0][AttributeKeys.IS_DYNAMIC] = dynamic_mask_list
        multi_inputs[0][AttributeKeys.GT_VELOCITY] = gt_velocity

        # Step 3: per-instance velocity from ICP on RoI point clouds.
        velocity = estimate_velocity_icp(
            multi_inputs=multi_inputs,
            dynamic_mask_list=dynamic_mask_list,
            min_points=self.config.min_points_for_icp,
            device=device,
        )
        for frame_key in multi_inputs:
            multi_inputs[frame_key][AttributeKeys.EST_VELOCITY] = velocity

        # Step 4: location and orientation from RoI LiDAR + velocity.
        multi_inputs = estimate_location_orientation(
            multi_inputs=multi_inputs,
            dynamic_mask_list=dynamic_mask_list,
            min_roi_points=self.config.min_roi_points,
            device=device,
        )

        # Optional: full GT attributes for debug / trainer loc fallback.
        if self.config.compute_gt:
            try:
                gt_location, gt_dimension, gt_orientation, gt_velocity, _ = compute_gt_attributes(
                    multi_inputs=multi_inputs,
                    dynamic_mask_list=dynamic_mask_list,
                    use_velocity_direction=True,
                    velocity_threshold=self.config.dynamic_velocity_threshold,
                )
                multi_inputs[0][AttributeKeys.GT_LOCATION] = gt_location
                multi_inputs[0][AttributeKeys.GT_DIMENSION] = gt_dimension
                multi_inputs[0][AttributeKeys.GT_ORIENTATION] = gt_orientation
                multi_inputs[0][AttributeKeys.GT_VELOCITY] = gt_velocity
            except Exception:
                multi_inputs[0][AttributeKeys.GT_LOCATION] = None
                multi_inputs[0][AttributeKeys.GT_DIMENSION] = None
                multi_inputs[0][AttributeKeys.GT_ORIENTATION] = None

        return multi_inputs


def estimate_initial_attributes(
    multi_inputs: dict,
    device: str = "cuda:0",
    *,
    compute_gt: bool = True,
    show_progress: bool = True,
    dynamic_velocity_threshold: float = DEFAULT_DYNAMIC_VELOCITY_THRESHOLD,
) -> dict:
    """Estimate initial location, velocity, and orientation for each instance."""
    config = InitialAttributesConfig(
        device=device,
        compute_gt=compute_gt,
        show_progress=show_progress,
        dynamic_velocity_threshold=dynamic_velocity_threshold,
    )
    return InitialAttributesPipeline(config).run(multi_inputs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test initial attribute pipeline.")
    parser.add_argument(
        "--input",
        type=str,
        default="Debug_Examples/exampleV2.pkl",
        help="Pickle with multi_inputs dict.",
    )
    args = parser.parse_args()

    inputs = read_pickle_file(args.input)
    outputs = estimate_initial_attributes(
        inputs,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    print("is_dynamic:", outputs[0][AttributeKeys.IS_DYNAMIC])
    print("est_location:", outputs[0][AttributeKeys.EST_LOCATION])
    print("est_velocity:", outputs[0][AttributeKeys.EST_VELOCITY])
    if outputs[0][AttributeKeys.GT_LOCATION] is not None:
        print("gt_location:", outputs[0][AttributeKeys.GT_LOCATION])
        print("gt_velocity:", outputs[0][AttributeKeys.GT_VELOCITY])
