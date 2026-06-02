"""Shared helpers for loading ``dynamic_mask.txt`` in validator tools."""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from preprocessing.Dynamic_Labels.format import parse_dynamic_line


def resolve_dataset_path(root_dirname: str, path: str) -> str:
    """Resolve a dataset-relative or absolute path under ``root_dirname``."""
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(root_dirname, path))


def dynamic_mask_path(dynamic_dirname: str, sequence: str) -> str:
    """Return ``{dynamic_dirname}/syncXX/dynamic_mask.txt`` for a sequence folder name."""
    sync_name = "sync" + sequence[-7:-5]
    return os.path.join(dynamic_dirname, sync_name, "dynamic_mask.txt")


def load_dynamic_mask_by_instance_ids(
    dynamic_dirname: str,
    sequence: str,
    dataset_root: str,
) -> dict[tuple[int, ...], list[int]]:
    """Map ``sampled_image_filenames`` instance-id tuples to dynamic flags."""
    mask_path = dynamic_mask_path(dynamic_dirname, sequence)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"dynamic_mask.txt not found: {mask_path}")

    result: dict[tuple[int, ...], list[int]] = {}
    with open(mask_path, encoding="utf-8") as file:
        for line in map(str.strip, file):
            if not line:
                continue
            instance_ids, _, labels = parse_dynamic_line(line, dataset_root)
            result[tuple(instance_ids)] = [int(label) for label in labels]
    return result


def load_dynamic_mask_by_image(
    dynamic_dirname: str,
    sequence: str,
    dataset_root: str,
) -> dict[str, tuple[list[int], list[float]]]:
    """Map absolute image paths to ``(instance_ids, dynamic_labels)``."""
    mask_path = dynamic_mask_path(dynamic_dirname, sequence)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"dynamic_mask.txt not found: {mask_path}")

    result: dict[str, tuple[list[int], list[float]]] = {}
    with open(mask_path, encoding="utf-8") as file:
        for line in map(str.strip, file):
            if not line:
                continue
            instance_ids, image_path, labels = parse_dynamic_line(line, dataset_root)
            result[image_path] = (instance_ids, [float(label) for label in labels])
    return result


def annotation_to_image_path(root_dirname: str, annotation_filename: str) -> str:
    """Convert ``annotations/.../frame.json`` to ``data_2d_raw/.../frame.png``."""
    rel_path = os.path.relpath(annotation_filename, root_dirname)
    image_rel = rel_path.replace("annotations/", "data_2d_raw/", 1)
    image_rel = os.path.splitext(image_rel)[0] + ".png"
    return resolve_dataset_path(root_dirname, image_rel)


def lookup_dynamic_labels(
    dynamic_by_image: dict[str, tuple[list[int], list[float]]],
    image_path: str,
    instance_ids: list,
) -> list[float] | None:
    """Return per-instance dynamic flags aligned to ``instance_ids``, or ``None`` if missing."""
    entry = dynamic_by_image.get(image_path)
    if entry is None:
        return None
    file_ids, file_labels = entry
    label_map = dict(zip(file_ids, file_labels))
    return [label_map.get(int(instance_id), 0.0) for instance_id in instance_ids]
