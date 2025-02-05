STAGE_1_Get_mAP(){

cd ../..

cd Metrics/Stage_1_Evaluation

pd_dir_folder="/media/zliu/data12/TPAMI_Results/TPAMI_Ablations/Velocity_Projection_Loss_Only/pseudo_labels_txt/predictions/"
gt_dir_folder="/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/training/label_gt/"
saved_mAP_folder="/media/zliu/data12/TPAMI_Results/TPAMI_Ablations/Velocity_Projection_Loss_Only/mAP"
trainlist="/media/zliu/data12/TPAMI_Results/TPAMI_Ablations/Velocity_Projection_Loss_Only/Sync_Files/training/sync_file.txt"

python mAP_statics.py --pd_dir_folder $pd_dir_folder \
                    --gt_dir_folder $gt_dir_folder \
                    --saved_mAP_folder $saved_mAP_folder \
                    --trainlist $trainlist

}


STAGE_1_Get_mAP