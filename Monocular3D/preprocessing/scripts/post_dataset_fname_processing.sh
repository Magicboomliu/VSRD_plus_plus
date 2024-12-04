UPDATE_2D_BBOX_WITH_INTERNIMAGE(){

cd ..
cd dataset_split/post_dataset_processing
root_dirname="/media/zliu/data12/dataset/TPAMI_Stage2/NEW_VSRDPP25_LABEL/VSRD_PP_SPLIT/"
det2d_path='/media/zliu/data12/dataset/VSRD_PP_Sync/det2d/'
threshold_option='threshold03'
output_folder="/media/zliu/data12/dataset/TPAMI_Stage2/NEW_VSRDPP25_LABEL/Replaced_Version/VSRD_PP_SPLIT_V2/"


python replace_2dbbox_with_intern_image.py     --root_dirname $root_dirname \
                                                --det2d_path $det2d_path \
                                                --threshold_option $threshold_option \
                                                --output_folder $output_folder




}

UPDATE_2D_BBOX_WITH_INTERNIMAGE