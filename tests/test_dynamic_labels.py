"""Tests for dynamic label file loading."""

import os

from trainer.utils.dynamic_labels import load_dynamic_labels_index, lookup_dynamic_mask


def test_load_and_lookup_dynamic_labels(tmp_path):
    root = str(tmp_path)
    img = os.path.join(root, "data_2d_raw/seq/image_00/data_rect/0001.png")
    dynamic_file = tmp_path / "dynamic_mask.txt"
    dynamic_file.write_text(
        "1,2 data_2d_raw/seq/image_00/data_rect/0001.png 0.0,1.0\n",
        encoding="utf-8",
    )

    index = load_dynamic_labels_index(str(dynamic_file), root)
    assert lookup_dynamic_mask(img, [1, 2], index) == [False, True]
    assert lookup_dynamic_mask(img, [99], index, default=False) == [False]
