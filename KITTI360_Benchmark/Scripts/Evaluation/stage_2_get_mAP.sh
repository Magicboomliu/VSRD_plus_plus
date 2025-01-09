STAGE_2_Get_mAP(){

cd ../..

cd Metrics/Stage_2_Evaluation

pd_dir_folder="/media/zliu/data12/TPAMI_Results/Stage2_Expermental_Results/Experimental_Results/Stage2/Static_Dynamic_Splits/MonoDETR/vsrd_PP/test/monodetr/outputs/data/"
gt_dir_folder="/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT/testing/label_gt/"
saved_mAP_folder="/media/zliu/data12/TPAMI_Results/Stage2_Expermental_Results/Experimental_Results/Stage2/Static_Dynamic_Splits/MonoDETR/vsrd_PP/mAP"

python mAP_statics_box2d.py --pd_dir_folder $pd_dir_folder \
                    --gt_dir_folder $gt_dir_folder \
                    --saved_mAP_folder $saved_mAP_folder

}


STAGE_2_Get_mAP