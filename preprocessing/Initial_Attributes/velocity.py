"""Estimate per-instance velocity via ICP on RoI LiDAR point clouds."""

import torch

from preprocessing.Initial_Attributes.keys import AttributeKeys
from preprocessing.Initial_Attributes.post_processing import icp_translation_only


def _drop_extreme_z_velocities(velocity_list: list[torch.Tensor]) -> list[int]:
    """Return indices of velocities to drop (min/max |z|)."""
    if len(velocity_list) <= 1:
        return []

    z_values = torch.tensor([velocity[0, 2].item() for velocity in velocity_list])
    z_abs = torch.abs(z_values)
    return [torch.argmax(z_abs).item(), torch.argmin(z_abs).item()]


def _icp_velocity_for_instance(
    frame_point_clouds: list[tuple[torch.Tensor, int]],
    *,
    is_static: bool,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Aggregate frame-to-frame ICP translations into one velocity vector."""
    frame_indices = [item[1] for item in frame_point_clouds]
    point_clouds = [item[0] for item in frame_point_clouds]

    velocity_list: list[torch.Tensor] = []
    point_counts: list[int] = []

    for idx in range(len(frame_indices) - 1):
        time_gap = frame_indices[idx + 1] - frame_indices[idx]
        if time_gap <= 0 or time_gap > 15:
            continue

        translation = icp_translation_only(
            A=point_clouds[idx].cpu().numpy(),
            B=point_clouds[idx + 1].cpu().numpy(),
        )
        velocity = torch.from_numpy(translation).to(point_clouds[idx].device) / time_gap
        velocity_list.append(velocity)
        point_counts.append(point_clouds[idx].shape[0])

    if not velocity_list:
        return torch.zeros((1, 3), device=device)

    drop_indices = _drop_extreme_z_velocities(velocity_list)
    weighted_sum = torch.zeros_like(velocity_list[0])
    total_points = 0

    for idx, velocity in enumerate(velocity_list):
        if idx in drop_indices:
            continue
        weighted_sum = weighted_sum + velocity * point_counts[idx]
        total_points += point_counts[idx]

    if total_points == 0:
        mean_velocity = torch.zeros((1, 3), device=device)
    else:
        mean_velocity = weighted_sum / total_points

    if is_static:
        if torch.max(mean_velocity) < 0.5:
            mean_velocity = mean_velocity * 0.1
    else:
        if mean_velocity[0][0] > 0.02:
            mean_velocity[0][0] = torch.tensor(0.02, device=mean_velocity.device) + 0.01 * mean_velocity[0][0]
        if mean_velocity[0][1] > 0.02:
            mean_velocity[0][1] = torch.tensor(0.02, device=mean_velocity.device) + 0.01 * mean_velocity[0][1]

    return mean_velocity


def estimate_velocity_icp(
    multi_inputs: dict,
    dynamic_mask_list: list[bool] | None = None,
    *,
    min_points: int = 100,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Estimate per-instance 3D velocity from multi-frame RoI LiDAR."""
    target_instance_ids = multi_inputs[0]["instance_ids"][0].cpu().numpy().tolist()
    frame_point_clouds: dict[int, list[tuple[torch.Tensor, int]]] = {
        instance_id: [] for instance_id in target_instance_ids
    }

    for frame_key, frame_inputs in multi_inputs.items():
        roi_lidar = frame_inputs[AttributeKeys.ROI_LIDAR]
        for instance_id, points in roi_lidar.items():
            if points is not None and points.shape[0] > min_points:
                frame_point_clouds[instance_id].append((points, int(frame_key)))

    velocity_list = []
    for idx, instance_id in enumerate(target_instance_ids):
        is_dynamic = dynamic_mask_list[idx] if dynamic_mask_list is not None else False
        velocity = _icp_velocity_for_instance(
            frame_point_clouds[instance_id],
            is_static=not is_dynamic,
            device=device,
        )
        velocity_list.append(velocity.to(device))

    return torch.cat(velocity_list, dim=0)
