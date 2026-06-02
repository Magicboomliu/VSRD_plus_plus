"""Read/write ``dynamic_mask.txt`` lines."""

from __future__ import annotations

import os


def parse_filenames_line(
    line: str,
    dataset_root: str = "",
) -> tuple[list[int], str, str, list[int]]:
    """Parse ``sampled_image_filenames.txt`` → ids, abs path, rel path, source rel indices."""
    ids_str, img_path, src_str = line.strip().split(" ")
    instance_ids = [int(x) for x in ids_str.split(",")]
    rel_path = img_path
    if dataset_root and os.path.isabs(img_path):
        rel_path = os.path.relpath(img_path, dataset_root)
    elif dataset_root and not os.path.isabs(img_path):
        img_path = os.path.join(dataset_root, img_path)
    source_indices = list(map(int, src_str.split(",")))
    return instance_ids, os.path.abspath(img_path), rel_path, source_indices


def parse_dynamic_line(
    line: str,
    dataset_root: str = "",
) -> tuple[list[int], str, list[bool]]:
    """Parse ``dynamic_mask.txt`` → instance ids, absolute image path, labels."""
    ids_str, img_path, labels_str = line.strip().split(" ")
    instance_ids = [int(x) for x in ids_str.split(",")]
    if dataset_root and not os.path.isabs(img_path):
        img_path = os.path.join(dataset_root, img_path)
    labels = [bool(int(float(x))) for x in labels_str.split(",")]
    return instance_ids, os.path.abspath(img_path), labels


def format_dynamic_line(
    instance_ids: list[int],
    image_path: str,
    is_dynamic: list[bool],
) -> str:
    """Format one output line matching legacy ``dynamic_mask.txt``."""
    ids_str = ",".join(str(i) for i in instance_ids)
    labels_str = ",".join("1.0" if flag else "0.0" for flag in is_dynamic)
    return f"{ids_str} {image_path} {labels_str}"


def load_sample_lines(
    filenames_path: str,
    dynamic_path: str,
    dataset_root: str,
):
    """Yield ``(target_img, source_rels, legacy_ids, legacy_labels)`` per target frame."""
    legacy_by_img: dict[str, tuple[list[int], list[bool]]] = {}
    with open(dynamic_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids, img_abs, labels = parse_dynamic_line(line, dataset_root)
            legacy_by_img[img_abs] = (ids, labels)

    with open(filenames_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, img_abs, _, src_rels = parse_filenames_line(line, dataset_root)
            legacy = legacy_by_img.get(img_abs)
            if legacy is None:
                continue
            legacy_ids, legacy_labels = legacy
            yield img_abs, src_rels, legacy_ids, legacy_labels
