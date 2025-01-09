STAGE_1_Get_mAP(){

cd ../..

cd Metrics/Stage_1_Evaluation

pd_dir_folder="/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/training/label_2/"
gt_dir_folder="/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/training/label_gt/"
saved_mAP_folder="/home/zliu/TPAMI25/Experimental_Results/Stage1/Static_Dynamic_Splits/mAP/VSRD"
trainlist="/media/zliu/data12/dataset/TPAMI_Stage2/NEW_VSRDPP25_LABEL/VSRD_PP_SPLIT/ImageSets/train.txt"

python mAP_statics.py --pd_dir_folder $pd_dir_folder \
                    --gt_dir_folder $gt_dir_folder \
                    --saved_mAP_folder $saved_mAP_folder \
                    --trainlist $trainlist

}


STAGE_1_Get_mAP