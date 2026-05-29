import os
import sys
import re


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines



def read_complex_strings(string_input, dataset_root=""):
    """Parse a dynamic_mask.txt line: '<ids> <path> <labels>'

    Works with both absolute paths and relative paths.  When dataset_root is
    provided and the path is relative, it is resolved to an absolute path so
    that callers can compare it directly with os.path.abspath image filenames.
    """
    pattern = r"([\d,]+)\s+(\S+)\s+([\d\.,\-]+)"
    match_groups = re.match(pattern, string_input)

    ids = match_groups.group(1)
    filename_path = match_groups.group(2)
    labels = match_groups.group(3)

    if dataset_root and not os.path.isabs(filename_path):
        filename_path = os.path.join(dataset_root, filename_path)

    return_dict = dict()
    return_dict['instance_ids'] = ids
    return_dict['filename'] = filename_path
    return_dict['labels'] = labels

    return return_dict
