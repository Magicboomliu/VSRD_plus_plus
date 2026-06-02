"""Initial 3D attribute estimation from pseudo depth and instance masks."""

from preprocessing.Initial_Attributes.extract import (
    InitialAttributesResult,
    extract_initial_attributes,
)
from preprocessing.Initial_Attributes.gt_attributes import (
    DEFAULT_DYNAMIC_VELOCITY_THRESHOLD,
    compute_gt_attributes,
    compute_gt_velocity,
    infer_dynamic_mask_from_gt_velocity,
    infer_dynamic_mask_from_multi_inputs,
)
from preprocessing.Initial_Attributes.keys import AttributeKeys
from preprocessing.Initial_Attributes.pipeline import (
    InitialAttributesConfig,
    InitialAttributesPipeline,
    attach_pseudo_depth,
    estimate_initial_attributes,
)

__all__ = [
    "AttributeKeys",
    "DEFAULT_DYNAMIC_VELOCITY_THRESHOLD",
    "InitialAttributesConfig",
    "InitialAttributesPipeline",
    "InitialAttributesResult",
    "attach_pseudo_depth",
    "compute_gt_attributes",
    "compute_gt_velocity",
    "estimate_initial_attributes",
    "extract_initial_attributes",
    "infer_dynamic_mask_from_gt_velocity",
    "infer_dynamic_mask_from_multi_inputs",
]
