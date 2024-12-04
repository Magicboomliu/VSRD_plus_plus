RANDOM_UPDATE_ESTIMATE_LABEL(){
estimated_label="/media/zliu/data12/dataset/TPAMI_Stage2/NEW_VSRDPP25_LABEL/Replaced_Version/VSRD_SPLIT_V2/training/label_2/"
gt_labels="/media/zliu/data12/dataset/TPAMI_Stage2/NEW_VSRDPP25_LABEL/Replaced_Version/VSRD_SPLIT_V2/training/label_gt/"
ratio=0.05

cd ..
cd dataset_split/post_dataset_processing/
python random_update_label.py --estimated_label $estimated_label \
                              --gt_labels $gt_labels \
                              --ratio $ratio




}

RANDOM_UPDATE_ESTIMATE_LABEL