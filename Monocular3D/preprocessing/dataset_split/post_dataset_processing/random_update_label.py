import os
import random
from tqdm import tqdm
import shutil
import argparse

if __name__=="__main__":
    

    parser = argparse.ArgumentParser(description="Convert VSRD Format into KITTI3D Dataset Configuration")
    parser.add_argument("--estimated_label", type=str, default="datasets/KITTI-360")
    parser.add_argument("--gt_labels",type=str, default="/data1/liu/VSRD_PP_Sync/det2d")
    parser.add_argument("--ratio",type=float,default=0.05)
    args = parser.parse_args()
    
    estimated_label = args.estimated_label
    gt_labels = args.gt_labels
    ratio = args.ratio
    
    cnt = 0
    for fname in tqdm(os.listdir(gt_labels)):

        gt_label_name = os.path.join(gt_labels,fname)
        est_label_name = gt_label_name.replace(gt_labels,estimated_label)
        
        assert os.path.exists(gt_label_name)
        assert os.path.exists(est_label_name)
        
        a = random.random()
        if a<ratio:
            cnt = cnt +1
            shutil.copy(gt_label_name,est_label_name)
    
    print(cnt/len(os.listdir(gt_labels)))
    print("Ok")


        