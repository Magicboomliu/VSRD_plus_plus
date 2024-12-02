import os
import numpy as np
from tqdm import tqdm
import shutil
import argparse
from scipy.optimize import linear_sum_assignment
import os

def save_bounding_boxes_with_type(filename, data, type_column='Car', fmt='%.6f'):
    """
    保存 (N, 16) 的 numpy 数组为文本文件，并在每行添加一个字符串类型列（如 'Car'）。

    参数:
        filename (str): 保存的文件名或路径。
        data (np.ndarray): 输入的 (N, 16) 数组。
        type_column (str or list): 每行的类型列，可以是单一字符串或字符串列表。
        fmt (str): 浮点数格式（默认保留 6 位小数，如 '%.6f'）。
    """
    if not isinstance(data, np.ndarray) or data.shape[1] != 15:
        raise ValueError("输入数据必须是 (N, 15) 的 numpy 数组。")
    
    # 确保 type_column 是一个字符串数组
    if isinstance(type_column, str):
        type_column = np.array([type_column] * data.shape[0])[:, None]
    elif isinstance(type_column, list):
        type_column = np.array(type_column)[:, None]
        if len(type_column) != data.shape[0]:
            raise ValueError("type_column 列表长度必须与数据行数相同。")
    else:
        raise TypeError("type_column 必须是字符串或字符串列表。")
    
    # 拼接类型列和数据
    data_with_type = np.hstack([type_column, data])

    # 保存到文件
    np.savetxt(
        filename, 
        data_with_type, 
        fmt='%s ' + (fmt + ' ') * (data.shape[1] - 1),  # 第一列字符串，后15列为浮点数
        delimiter=' '  # 使用空格分隔
    )

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def starts_with_2013(string):
    start_index = string.find("2013")
    return string[start_index:]

def saved_into_txt(saved_name,content_list):
    with open(saved_name,'w') as f:
        for idx, content in enumerate(content_list):
            if idx!=len(content_list)-1:
                f.writelines(content+"\n")
            else:
                f.writelines(content)

def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    box1: [x_min, y_min, x_max, y_max]
    box2: [x_min, y_min, x_max, y_max]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def bilateral_matching(boxes1, boxes2):
    """
    Perform bilateral matching using IoU.
    boxes1: numpy array of shape [N1, 4]
    boxes2: numpy array of shape [N2, 4]
    Returns:
        index1: indices of boxes1 that match with boxes2
        index2: indices of boxes2 that match with boxes1
    """
    N1, N2 = len(boxes1), len(boxes2)
    iou_matrix = np.zeros((N1, N2))

    # Compute IoU for all pairs
    for i in range(N1):
        for j in range(N2):
            iou_matrix[i, j] = compute_iou(boxes1[i], boxes2[j])

    # Match from boxes1 to boxes2
    match1_to_2 = np.argmax(iou_matrix, axis=1)
    match2_to_1 = np.argmax(iou_matrix, axis=0)

    # Perform bilateral matching
    matched_indices1 = []
    matched_indices2 = []

    for i in range(N1):
        j = match1_to_2[i]
        if match2_to_1[j] == i:  # Ensure bilateral match
            matched_indices1.append(i)
            matched_indices2.append(j)

    return np.array(matched_indices1), np.array(matched_indices2)


def main(args):
    
    output_folder = args.output_folder
    os.makedirs(output_folder,exist_ok=True)

    # new training folder creation
    output_training_folder = os.path.join(output_folder,'training')
    os.makedirs(output_training_folder,exist_ok=True)
    
    output_training_label_2_folder = os.path.join(output_folder,'training','label_2')
    output_training_label_gt_folder = os.path.join(output_folder,'training','label_gt')
    output_training_calib_folder = os.path.join(output_folder,'training','calib')
    output_training_sync_file = os.path.join(output_folder,'training','sync_file.txt')
    
    output_training_image_2_folder = os.path.join(output_folder,'training','image_2')
    output_training_image_3_folder = os.path.join(output_folder,'training','image_3')
    
    
    output_ImageSets_folder = os.path.join(output_folder,"ImageSets")
    source_ImageSets_folder = os.path.join(args.root_dirname,"ImageSets")
    assert os.path.exists(source_ImageSets_folder)
    
    if not os.path.exists(output_ImageSets_folder):
        os.system("cp -r {} {}".format(source_ImageSets_folder,output_ImageSets_folder))

    
    if not os.path.exists(output_training_label_gt_folder):
        os.system("ln -s {} {}".format(output_training_label_gt_folder.replace(args.output_folder,args.root_dirname),
                                                                                output_training_label_gt_folder))
        
    if not os.path.exists(output_training_calib_folder):
        os.system("ln -s {} {}".format(output_training_calib_folder.replace(args.output_folder,args.root_dirname),
                                                                                output_training_calib_folder))
        
    if not os.path.exists(output_training_image_2_folder):
        os.system("ln -s {} {}".format(output_training_image_2_folder.replace(args.output_folder,args.root_dirname),
                                                                                output_training_image_2_folder))
        
    if not os.path.exists(output_training_image_3_folder):
        os.system("ln -s {} {}".format(output_training_image_3_folder.replace(args.output_folder,args.root_dirname),
                                                                                output_training_image_3_folder))
        
    if not os.path.exists(output_training_sync_file):
        shutil.copy(output_training_sync_file.replace(args.output_folder,args.root_dirname),output_training_sync_file)
    
    os.makedirs(output_training_label_2_folder,exist_ok=True) # label 2 : need to be optimized
    


    
    # new testing folder creation
    output_testing_folder = os.path.join(output_folder,'testing')
    os.makedirs(output_testing_folder,exist_ok=True)
    output_testing_label_2_folder = os.path.join(output_folder,'testing','label_2')
    output_testing_label_gt_folder = os.path.join(output_folder,'testing','label_gt')
    output_testing_calib_folder = os.path.join(output_folder,'testing','calib')
    output_testing_sync_file = os.path.join(output_folder,'testing','sync_file.txt')

    output_testing_image_2_folder = os.path.join(output_folder,'testing','image_2')
    output_testing_image_3_folder = os.path.join(output_folder,'testing','image_3')


    if not os.path.exists(output_testing_label_gt_folder):
        os.system("ln -s {} {}".format(output_testing_label_gt_folder.replace(args.output_folder,args.root_dirname),
                                                                                output_testing_label_gt_folder))
    
    if not os.path.exists(output_testing_calib_folder):
        os.system("ln -s {} {}".format(output_testing_calib_folder.replace(args.output_folder,args.root_dirname),
                                                                                output_testing_calib_folder))
    
    if not os.path.exists(output_testing_image_2_folder):
        os.system("ln -s {} {}".format(output_testing_image_2_folder.replace(args.output_folder,args.root_dirname),
                                                                                output_testing_image_2_folder))
    
    if not os.path.exists(output_testing_image_3_folder):
        os.system("ln -s {} {}".format(output_testing_image_3_folder.replace(args.output_folder,args.root_dirname),
                                                                                output_testing_image_3_folder))
    
    
    if not os.path.exists(output_testing_sync_file):
        shutil.copy(output_testing_sync_file.replace(args.output_folder,args.root_dirname),output_testing_sync_file)
    os.makedirs(output_testing_label_2_folder,exist_ok=True)

    
    training_validation_sync_files = os.path.join(args.root_dirname,"training","sync_file.txt")
    testing_sync_files =  os.path.join(args.root_dirname,"testing","sync_file.txt")
    assert os.path.exists(training_validation_sync_files)
    assert os.path.exists(testing_sync_files)
    boxes_2d_path = os.path.join(args.det2d_path,args.threshold_option)
    assert os.path.exists(boxes_2d_path)
    


    ''' Train and Validation Files Creation'''
    trainval_contents = read_text_lines(training_validation_sync_files)
    for line in tqdm(trainval_contents):
        source_image_filename_left,saved_image_left, _,_,_,_= line.split(" ")
        basename = os.path.basename(saved_image_left)[:-4]
        seq_image_name = starts_with_2013(source_image_filename_left)
        internimage_2d_file = os.path.join(boxes_2d_path,seq_image_name).replace(".png",'.txt')
        est_label = saved_image_left.replace("image_2",'label_2').replace(".png",'.txt')
        assert os.path.exists(est_label)
        
        updated_est_label_fname = est_label.replace(args.root_dirname,args.output_folder)

        
        if os.path.exists(internimage_2d_file):
            vsrd_pp_vanilla_label = np.loadtxt(est_label, dtype=str).reshape(-1, 16)
            internimage_2d_label = np.loadtxt(internimage_2d_file,dtype=str).reshape(-1,6)
            
            vsrd_projected2d_box = vsrd_pp_vanilla_label[:,4:8].astype(np.float32)
            internimage_2d_box = internimage_2d_label[:,1:5].astype(np.float32)
            
            index_1, index_2 = bilateral_matching(boxes1=vsrd_projected2d_box,boxes2=internimage_2d_box)            
            vsrd_projected2d_box = vsrd_projected2d_box[index_1]
            internimage_2d_box = internimage_2d_box[index_2]
            
            vsrd_pp_vanilla_label = vsrd_pp_vanilla_label[index_1]
            internimage_2d_label = internimage_2d_label[index_2]
            vsrd_pp_vanilla_label[:,4:8] = internimage_2d_label[:,1:5]
            

        else:
            # keep the orginal estimated label
            pass
        
        np.savetxt(updated_est_label_fname,
                    vsrd_pp_vanilla_label,
                    fmt='%s ' * 16,  
                    delimiter=' ' )


    ''' Testing Files Creation'''
    trainval_contents = read_text_lines(testing_sync_files)
    for line in tqdm(trainval_contents):
        source_image_filename_left,saved_image_left, _,_,_,_= line.split(" ")
        basename = os.path.basename(saved_image_left)[:-4]
        seq_image_name = starts_with_2013(source_image_filename_left)
        internimage_2d_file = os.path.join(boxes_2d_path,seq_image_name).replace(".png",'.txt')
        est_label = saved_image_left.replace("image_2",'label_2').replace(".png",'.txt')
        assert os.path.exists(est_label)        
        updated_est_label_fname = est_label.replace(args.root_dirname,args.output_folder)

        if os.path.exists(internimage_2d_file):
            vsrd_pp_vanilla_label = np.loadtxt(est_label, dtype=str).reshape(-1, 16)
            internimage_2d_label = np.loadtxt(internimage_2d_file,dtype=str).reshape(-1,6)
            
            vsrd_projected2d_box = vsrd_pp_vanilla_label[:,4:8].astype(np.float32)
            internimage_2d_box = internimage_2d_label[:,1:5].astype(np.float32)
            
            index_1, index_2 = bilateral_matching(boxes1=vsrd_projected2d_box,boxes2=internimage_2d_box)            
            vsrd_projected2d_box = vsrd_projected2d_box[index_1]
            internimage_2d_box = internimage_2d_box[index_2]
            
            vsrd_pp_vanilla_label = vsrd_pp_vanilla_label[index_1]
            internimage_2d_label = internimage_2d_label[index_2]
            vsrd_pp_vanilla_label[:,4:8] = internimage_2d_label[:,1:5]
            
        else:
            # keep the orginal estimated label
            pass
        
        np.savetxt(updated_est_label_fname,
                    vsrd_pp_vanilla_label,
                    fmt='%s ' * 16,  
                    delimiter=' ' )



if __name__=="__main__":


    parser = argparse.ArgumentParser(description="Convert VSRD Format into KITTI3D Dataset Configuration")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-360")
    parser.add_argument("--det2d_path",type=str, default="/data1/liu/VSRD_PP_Sync/det2d")
    parser.add_argument("--threshold_option",type=str,default="datasets/KITTI-360") # selected from 'threshold03' and 'threshold05'
    parser.add_argument("--output_folder",type=str,default='daatset')
    
    args = parser.parse_args()
    
    main(args=args)