import os
import sys
import re


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines



def read_complex_strings(string_input):
    
    pattern = r"([\d,\s]+)\s+(/media[^\s]+)\s+([\d\.,\s]+)"
    match_groups = re.match(pattern, string_input)

    ids = match_groups.group(1)  
    filename_path = match_groups.group(2)
    labels = match_groups.group(3)  
    
    return_dict = dict()
    
    return_dict['instance_ids'] = ids
    return_dict['filename'] = filename_path
    return_dict['labels'] = labels
    
    
    return  return_dict
