GET_mAP(){
cd ..
cd tools/IoU_AP_Analysis/Stage1_Eval

pd_dir_folder="/home/Journals2025/VSRD_plus_plus/Round1/Seg_Ablations/prefect_seg/training/label_est"
gt_dir_folder="/home/Journals2025/VSRD_plus_plus/Round1/Seg_Ablations/prefect_seg/training/label_2"
saved_mAP_folder="/home/Journals2025/VSRD_plus_plus/Round1/Seg_Ablations/prefect_seg/training/mAP"

python get_mAP.py --pd_dir_folder $pd_dir_folder \
                 --gt_dir_folder $gt_dir_folder \
                 --saved_mAP_folder $saved_mAP_folder

}


GET_mAP