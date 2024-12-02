RANDOM_UPDATE_ESTIMATE_LABEL(){
estimated_label="/data3/PAMI_Datasets/Replace_Version/VSRD_PP_SPLIT_Ver2/training/label_2/"
gt_labels="/data3/PAMI_Datasets/Replace_Version/VSRD_PP_SPLIT_Ver1/training/label_gt/"
ratio=0.05

cd ..
cd dataset_split/post_dataset_processing/
python random_update_label.py --estimated_label $estimated_label \
                              --gt_labels $gt_labels \
                              --ratio $ratio




}

RANDOM_UPDATE_ESTIMATE_LABEL