STAGE_2_Get_mAP(){

cd ../..

cd Metrics/Stage_2_Evaluation

pd_dir_folder="/media/zliu/data12/TPAMI_Results/TPAMI_Ablations/Synced_For_Evaluations/VSRD_Proj_And_Render/training/label_2/"
gt_dir_folder="/media/zliu/data12/TPAMI_Results/TPAMI_Ablations/Synced_For_Evaluations/VSRD_Proj_And_Render/training/label_gt/"
saved_mAP_folder="/media/zliu/data12/TPAMI_Results/TPAMI_Ablations/Velocity_Project_and_Sil_Loss/mAP"

python mAP_statics_box2d.py --pd_dir_folder $pd_dir_folder \
                    --gt_dir_folder $gt_dir_folder \
                    --saved_mAP_folder $saved_mAP_folder

}


STAGE_2_Get_mAP