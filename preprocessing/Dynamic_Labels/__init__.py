"""Offline generation of per-instance dynamic labels from GT bbox velocity."""

from preprocessing.Dynamic_Labels.format import (
    format_dynamic_line,
    load_sample_lines,
    parse_dynamic_line,
    parse_filenames_line,
)
from preprocessing.Dynamic_Labels.multi_inputs import build_multi_from_annotations

__all__ = [
    "DynamicLabelsConfig",
    "DynamicLabelsPipeline",
    "DynamicLabelsResult",
    "build_multi_from_annotations",
    "format_dynamic_line",
    "generate_dynamic_labels",
    "load_sample_lines",
    "parse_dynamic_line",
    "parse_filenames_line",
]


def __getattr__(name: str):
    if name in ("DynamicLabelsConfig", "DynamicLabelsPipeline", "DynamicLabelsResult", "generate_dynamic_labels"):
        from preprocessing.Dynamic_Labels import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
