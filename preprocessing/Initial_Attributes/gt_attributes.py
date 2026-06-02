"""Ground-truth 3D attributes derived from annotated bounding boxes."""

from __future__ import annotations

import torch
import torch.nn as nn

from preprocessing.Initial_Attributes.geometry_op import (
    encode_box_3d,
    rotation_matrix_y,
    transform_bounding_boxes_to_world,
)

# Dynamic if ||v|| >= threshold (m per relative frame index).
# Tuned on all 9 KITTI360 sequences vs legacy dynamic_mask.txt (44k instances, recall=100%):
# 0.20 m/frame → 99.98% accuracy, 11 FP, 0 FN. See scripts/sweep_dynamic_threshold_global.py.
DEFAULT_DYNAMIC_VELOCITY_THRESHOLD = 0.20


def _mean_tensor(tensors: list[torch.Tensor]) -> torch.Tensor:
    mean = torch.zeros_like(tensors[0])
    for tensor in tensors:
        mean += tensor
    return mean / len(tensors)


def _gt_velocity_from_boxes(multi_inputs: dict) -> tuple[dict, dict]:
    """Compute per-instance GT velocity from world-space bbox centres."""
    target_ids = multi_inputs[0]["instance_ids"][0].cpu().numpy().tolist()
    velocity_accum = {instance_id: [] for instance_id in target_ids}
    velocity_mean: dict[int, torch.Tensor] = {}

    frame_keys = sorted(multi_inputs.keys())
    for idx in range(len(frame_keys) - 1):
        current_key = frame_keys[idx]
        next_key = frame_keys[idx + 1]

        current_inputs = multi_inputs[current_key]
        next_inputs = multi_inputs[next_key]

        world_current = transform_bounding_boxes_to_world(
            extrinsic_matrix=torch.inverse(current_inputs["extrinsic_matrices"]),
            bounding_boxes_camera=current_inputs["boxes_3d"][0],
        )
        world_next = transform_bounding_boxes_to_world(
            extrinsic_matrix=torch.inverse(next_inputs["extrinsic_matrices"]),
            bounding_boxes_camera=next_inputs["boxes_3d"][0],
        )

        visible = current_inputs["visible_masks"][0] * next_inputs["visible_masks"][0]
        time_gap = next_key - current_key
        displacement = world_next.mean(dim=1) - world_current.mean(dim=1)

        for sub_idx, instance_id in enumerate(target_ids):
            if visible[sub_idx]:
                velocity_accum[instance_id].append(displacement[sub_idx] / time_gap)

    device = multi_inputs[0]["boxes_3d"][0].device
    for instance_id, values in velocity_accum.items():
        if values:
            velocity_mean[instance_id] = _mean_tensor(values)
        else:
            velocity_mean[instance_id] = torch.zeros(3, device=device)

    return velocity_accum, velocity_mean


def compute_gt_velocity(multi_inputs: dict) -> torch.Tensor:
    """Return GT velocity tensor ``[1, N, 3]`` in m per relative frame index."""
    target_ids = multi_inputs[0]["instance_ids"][0].cpu().numpy().tolist()
    _, velocity_mean = _gt_velocity_from_boxes(multi_inputs)
    velocity_list = [velocity_mean[instance_id].unsqueeze(0) for instance_id in target_ids]
    return torch.cat(velocity_list, dim=0).unsqueeze(0)


def infer_dynamic_mask_from_gt_velocity(
    gt_velocity: torch.Tensor,
    threshold: float = DEFAULT_DYNAMIC_VELOCITY_THRESHOLD,
) -> list[bool]:
    """Classify instances as dynamic when ``||v|| >= threshold`` (m/frame)."""
    speeds = gt_velocity.squeeze(0).norm(dim=-1)
    return [speed.item() >= threshold for speed in speeds]


def infer_dynamic_mask_from_multi_inputs(
    multi_inputs: dict,
    threshold: float = DEFAULT_DYNAMIC_VELOCITY_THRESHOLD,
) -> list[bool]:
    """Infer per-instance dynamic flags from annotation 3D boxes."""
    gt_velocity = compute_gt_velocity(multi_inputs)
    return infer_dynamic_mask_from_gt_velocity(gt_velocity, threshold=threshold)


def compute_gt_attributes(
    multi_inputs: dict,
    *,
    dynamic_mask_list: list[bool] | None = None,
    use_velocity_direction: bool = False,
    velocity_threshold: float = DEFAULT_DYNAMIC_VELOCITY_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[bool]]:
    """Return GT location, dimension, orientation, velocity, and inferred dynamic mask."""
    target_inputs = multi_inputs[0]

    boxes_world = transform_bounding_boxes_to_world(
        extrinsic_matrix=torch.inverse(target_inputs["extrinsic_matrices"]),
        bounding_boxes_camera=target_inputs["boxes_3d"][0],
    )
    locations, dimensions, orientations = encode_box_3d(boxes_world.unsqueeze(0))
    gt_velocity = compute_gt_velocity(multi_inputs)

    if dynamic_mask_list is None:
        dynamic_mask_list = infer_dynamic_mask_from_gt_velocity(
            gt_velocity,
            threshold=velocity_threshold,
        )

    if use_velocity_direction:
        orientations_from_velocity = nn.functional.normalize(gt_velocity[..., [2, 0]], dim=-1)
        orientations_from_velocity = rotation_matrix_y(
            *torch.unbind(orientations_from_velocity, dim=-1)
        )
        dynamic = torch.tensor(dynamic_mask_list, dtype=torch.bool).view(1, -1, 1, 1).to(gt_velocity.device)
        orientations = torch.where(dynamic, orientations_from_velocity, orientations)

    return locations, dimensions, orientations, gt_velocity, dynamic_mask_list


get_gt_location_velocity_orientation = compute_gt_attributes
