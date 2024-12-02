SPLIT_THE_FNAME_BY_SEQUENCE(){

cd ..
cd dataset_split/pre_dataset_split

root_dirname="/data3/VSRD_PP_Sync/"
pseudo_labels_dirname="/data3/TPAMI_STAGE2/VSRDPP_Tsubame_Results/VSRDPP_Pesudo_Labels/"
training_split="00,02,04,05,06,09"
valiation_split="03,07"
testing_split="10"
output_dataset_path="/data3/PAMI_Datasets/VSRD_SPLIT"
output_dataset_splits_filename_path="/data3/PAMI_Datasets/VSRD_SPLIT"


python split_by_sequences.py    --pseudo_labels_dirname $pseudo_labels_dirname \
                                --root_dirname $root_dirname \
                                --training_split $training_split \
                                --valiation_split $valiation_split \
                                --testing_split $testing_split \
                                --output_dataset_path $output_dataset_path \
                                --output_dataset_splits_filename_path $output_dataset_splits_filename_path


}


SPLIT_THE_FNAME_BY_DYNAMIC(){

cd ..
cd dataset_split/pre_dataset_split

root_dirname="/data3/VSRD_PP_Sync/"
pseudo_labels_dirname="/data3/TPAMI_STAGE2/VSRDPP_Tsubame_Results/VSRDPP_Pesudo_Labels/"
dynamic_static_ratio=0.25
trainval_ratio=0.5
testing_split="10"
output_dataset_path="/data3/PAMI_Datasets/VSRD_PP_SPLIT"
output_dataset_splits_filename_path="/data3/PAMI_Datasets/VSRD_PP_SPLIT"


python split_by_dynamic_static.py --root_dirname $root_dirname \
                                    --pseudo_labels_dirname $pseudo_labels_dirname \
                                    --dynamic_static_ratio $dynamic_static_ratio \
                                    --testing_split $testing_split \
                                    --output_dataset_path $output_dataset_path \
                                    --output_dataset_splits_filename_path $output_dataset_splits_filename_path \
                                    --trainval_ratio $trainval_ratio
}


# Split the train/val/test by the sequence
# SPLIT_THE_FNAME_BY_SEQUENCE


SPLIT_THE_FNAME_BY_DYNAMIC