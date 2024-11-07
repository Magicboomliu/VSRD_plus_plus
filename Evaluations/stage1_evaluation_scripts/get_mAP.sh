GET_mAP(){
cd ..
cd tools/IoU_AP_Analysis/Stage1_Eval

pd_dir_folder="/media/zliu/data12/dataset/KITTI360_Tsubame_ALL/training/label_est/"
gt_dir_folder="/media/zliu/data12/dataset/KITTI360_Tsubame_ALL/training/label_2/"
saved_mAP_folder="/media/zliu/data12/dataset/KITTI360_Tsubame_ALL/training/mAP_Results"

python get_mAP.py --pd_dir_folder $pd_dir_folder \
                 --gt_dir_folder $gt_dir_folder \
                 --saved_mAP_folder $saved_mAP_folder

}


GET_mAP