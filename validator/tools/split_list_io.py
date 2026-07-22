"""Parse merged Stage1 filename / dynamic lists for validator Step1."""

from __future__ import annotations

import os
from dataclasses import dataclass

from preprocessing.Dynamic_Labels.format import parse_dynamic_line
from validator.tools.dynamic_labels_io import resolve_dataset_path


@dataclass(frozen=True)
class FilenameListEntry:
    instance_ids: tuple[int, ...]
    target_image_abs: str
    target_image_rel: str
    source_image_paths: list[str]
    dynamic_labels: list[bool]


def parse_filenames_line(line: str, dataset_root: str) -> tuple[tuple[int, ...], str, str, list[int]]:
    ids_str, img_path, src_str = line.strip().split(" ")
    instance_ids = tuple(int(x) for x in ids_str.split(","))
    rel_path = img_path.replace("\\", "/")
    target_abs = resolve_dataset_path(dataset_root, rel_path)
    source_indices = list(map(int, src_str.split(",")))
    return instance_ids, target_abs, rel_path, source_indices


def source_paths_from_relative_indices(target_abs: str, source_relative_indices: list[int]) -> list[str]:
    target_dir = os.path.dirname(target_abs)
    frame_idx = int(os.path.splitext(os.path.basename(target_abs))[0])
    return [
        os.path.join(target_dir, f"{frame_idx + rel:010}.png")
        for rel in source_relative_indices
    ]


def load_dynamic_index(dynamic_path: str, dataset_root: str) -> dict[tuple[int, ...], list[bool]]:
    index: dict[tuple[int, ...], list[bool]] = {}
    with open(dynamic_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instance_ids, _, labels = parse_dynamic_line(line, dataset_root)
            index[tuple(instance_ids)] = [bool(x) for x in labels]
    return index


def load_filename_list_entries(filenames_path: str, dynamic_path: str, dataset_root: str) -> list[FilenameListEntry]:
    dynamic_index = load_dynamic_index(dynamic_path, dataset_root)
    entries: list[FilenameListEntry] = []
    with open(filenames_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instance_ids, target_abs, target_rel, source_indices = parse_filenames_line(line, dataset_root)
            if instance_ids not in dynamic_index:
                raise KeyError(f"Missing dynamic labels for instance ids {instance_ids} (target={target_rel})")
            entries.append(
                FilenameListEntry(
                    instance_ids=instance_ids,
                    target_image_abs=target_abs,
                    target_image_rel=target_rel,
                    source_image_paths=source_paths_from_relative_indices(target_abs, source_indices),
                    dynamic_labels=dynamic_index[instance_ids],
                )
            )
    return entries
