import os

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def split_list(input_list, num_parts=8):
    """
    Split a list into specified number of parts.

    Args:
        input_list (list): The list to be split.
        num_parts (int): Number of parts to split the list into.

    Returns:
        list: A list containing the split sublists.
    """
    # Calculate the size of each part
    avg_length = len(input_list) // num_parts
    remainder = len(input_list) % num_parts

    # Generate sublists
    sublists = []
    start = 0
    for i in range(num_parts):
        end = start + avg_length + (1 if i < remainder else 0)
        sublists.append(input_list[start:end])
        start = end

    return sublists

def save_into_filename(my_list,filename):
    with open(filename,'w') as f:
        for idx, line in enumerate(my_list):
            if idx!=len(my_list)-1:     
                f.writelines(line+"\n")
            else:
                f.writelines(line)



if __name__=="__main__":
    complete_filename = "/home/zliu/TPAMI25/AutoLabels/SDFlabel/data_preprocssing/all_filenames.txt"
    
    lines = read_text_lines(complete_filename)
    sublist = split_list(lines)
    
    
    for idx, lst in enumerate(sublist):
        saved_name = "/home/zliu/TPAMI25/AutoLabels/SDFlabel/data_preprocssing/Splits_Filenames/all_filenames_{}.txt".format(idx)
        save_into_filename(my_list=lst,filename=saved_name)
