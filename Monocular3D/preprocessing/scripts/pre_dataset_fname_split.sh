SPLIT_THE_FNAME_BY_SEQUENCE(){

cd ..
cd dataset_split/pre_dataset_split

root_dirname="/media/zliu/data12/dataset/VSRD_PP_Sync/"
pseudo_labels_dirname="/media/zliu/data12/TPAMI_Results/Stage1_Expermental_Results/VSRD_Vanilla"
training_split="00,02,04,05,06,09"
valiation_split="03,07"
testing_split="10"
output_dataset_path="/media/zliu/data12/dataset/TPAMI_Stage2/NEW_VSRDPP25_LABEL/VSRD_SPLIT"
output_dataset_splits_filename_path="/media/zliu/data12/dataset/TPAMI_Stage2/NEW_VSRDPP25_LABEL/VSRD_SPLIT"

python split_by_sequences.py    --pseudo_labels_dirname $pseudo_labels_dirname \
                                --root_dirname $root_dirname \
                                --training_split $training_split \
                                --valiation_split $valiation_split \
                                --testing_split $testing_split \
                                --output_dataset_path $output_dataset_path \
                                --output_dataset_splits_filename_path $output_dataset_splits_filename_path


}



# Only For the First Time 
SPLIT_THE_FNAME_BY_DYNAMIC(){

cd ..
cd dataset_split/pre_dataset_split

root_dirname="/data1/liu/VSRD_PP_Sync/"
pseudo_labels_dirname="/data1/liu/VSRDPP_Tsubame_Results/VSRDPP_Pesudo_Labels/"
dynamic_static_ratio=0.25
trainval_ratio=0.5
testing_split="10"
output_dataset_path="/data1/liu/TPAMI25_Stage2_Datasets/VSRD_PP_SPLIT"
output_dataset_splits_filename_path="/data1/liu/TPAMI25_Stage2_Datasets/VSRD_PP_SPLIT"


python split_by_dynamic_static.py --root_dirname $root_dirname \
                                    --pseudo_labels_dirname $pseudo_labels_dirname \
                                    --dynamic_static_ratio $dynamic_static_ratio \
                                    --testing_split $testing_split \
                                    --output_dataset_path $output_dataset_path \
                                    --output_dataset_splits_filename_path $output_dataset_splits_filename_path \
                                    --trainval_ratio $trainval_ratio
}



SPLIT_THE_FNAME_BY_DYNAMIC_From_SYNC_FILE(){

cd ..
cd dataset_split/pre_dataset_split

root_dirname="/media/zliu/data12/dataset/VSRD_PP_Sync/"
pseudo_labels_dirname="/media/zliu/data12/TPAMI_Results/Stage1_Expermental_Results/Autolabels_Re_Train/pseudo_labels_txt/"
output_dataset_path="/media/zliu/data12/TPAMI_Results/Stage1_Expermental_Results/Synced_For_Evaluations/Splits_2024/Autolabels_Re_Train"
output_dataset_splits_filename_path="/media/zliu/data12/TPAMI_Results/Stage1_Expermental_Results/Synced_For_Evaluations/Splits_2024/Autolabels_Re_Train"
sync_folder_path="/media/zliu/data12/TPAMI_Results/Stage1_Expermental_Results/Autolabels_Re_Train/Sync_Files/"
python split_dynamic_static_by_sync_file.py --root_dirname $root_dirname \
                                    --pseudo_labels_dirname $pseudo_labels_dirname \
                                    --output_dataset_path $output_dataset_path \
                                    --output_dataset_splits_filename_path $output_dataset_splits_filename_path \
                                    --sync_folder_path $sync_folder_path

}

SPLIT_THE_FNAME_BY_SEQUENCE
# SPLIT_THE_FNAME_BY_DYNAMIC_From_SYNC_FILE