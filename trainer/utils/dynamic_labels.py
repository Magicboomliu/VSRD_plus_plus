"""Load per-frame dynamic flags from ``dynamic_mask.txt`` for training."""

from __future__ import annotations

import os

from preprocessing.Dynamic_Labels.format import parse_dynamic_line


def load_dynamic_labels_index(dynamic_path: str, dataset_root: str = "") -> dict[str, dict[int, bool]]:
    """Map absolute image path → {instance_id: is_dynamic}."""
    index: dict[str, dict[int, bool]] = {}
    with open(dynamic_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instance_ids, image_path, labels = parse_dynamic_line(line, dataset_root)
            row = index.setdefault(os.path.abspath(image_path), {})
            for instance_id, is_dynamic in zip(instance_ids, labels):
                row[int(instance_id)] = bool(is_dynamic)
    return index


def lookup_dynamic_mask(
    image_path: str,
    instance_ids,
    index: dict[str, dict[int, bool]],
    *,
    default: bool = False,
) -> list[bool]:
    """Return dynamic flags aligned with ``instance_ids`` (target-view order)."""
    row = index.get(os.path.abspath(image_path), {})
    out: list[bool] = []
    for instance_id in instance_ids:
        iid = int(instance_id.item()) if hasattr(instance_id, "item") else int(instance_id)
        out.append(row.get(iid, default))
    return out
