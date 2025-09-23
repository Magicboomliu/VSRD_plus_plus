import os
from tqdm import tqdm 
import sys


def read_multifile_txt(txt_path):
    """
    读取每行多个路径的 txt 文件。
    返回: list，每个元素是一个 tuple，包含该行的所有路径。
    """
    samples = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()   # 默认按空格切分
            samples.append(tuple(parts))
    return samples



if __name__ == "__main__":
    
    # vsrd24 splits for training and the validation sets.
    
    vsrd24_splits_sync_training_path = "/data1/liu/PAMI_Datasets/Replaced_With_GT_Boxes/VSRD/Casual_Splits_2025/training/sync_file.txt"
    vsrd24_splits_sync_validation_path = "/data1/liu/PAMI_Datasets/Replaced_With_GT_Boxes/VSRD/Casual_Splits_2025/testing/sync_file.txt"
    

    data_contents = read_multifile_txt(vsrd24_splits_sync_validation_path)
    
    for data_content in tqdm(data_contents):
        source_image_path = data_content[0] # gt images
        target_image_path = data_content[1]
        target_image_path = target_image_path.replace("ALL_PSEUDO_LABELS", "Replaced_With_GT_Boxes")
        target_image_path = target_image_path.replace("/VSRD/", "/GT/")
        
        source_label_path = data_content[4] # gt labels
        target_label_path = data_content[5]
        target_label_path = target_label_path.replace("ALL_PSEUDO_LABELS", "Replaced_With_GT_Boxes")
        target_label_path = target_label_path.replace("/VSRD/", "/GT/").replace("label_gt", "label_2")

        assert os.path.exists(source_image_path), "source image path does not exist: " + source_image_path
        assert os.path.exists(source_label_path), "source label path does not exist: " + source_label_path

        os.makedirs(os.path.dirname(target_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(target_label_path), exist_ok=True)

        
        if not os.path.exists(target_image_path):
            os.system("ln -s {} {}".format(source_image_path, target_image_path))
        
        
        if not os.path.exists(target_label_path):
            os.system("ln -s {} {}".format(source_label_path, target_label_path))
        
        
        # create 
        source_calib_path = "/data1/liu/VSRD_PP_Sync/cam_calib.txt"
        target_calib_path = target_label_path.replace("label_2", "calib")
        
        os.makedirs(os.path.dirname(target_calib_path), exist_ok=True)
        
        if not os.path.exists(target_calib_path):
            os.system("ln -s {} {}".format(source_calib_path, target_calib_path))
