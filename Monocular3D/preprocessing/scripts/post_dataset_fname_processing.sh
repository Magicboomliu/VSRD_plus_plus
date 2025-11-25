UPDATE_2D_BBOX_WITH_INTERNIMAGE(){

cd ..
cd dataset_split/post_dataset_processing
root_dirname="/data3/PAMI_Datasets/VSRD_PP_SPLIT/"
det2d_path='/data3/VSRD_PP_Sync/det2d/'
threshold_option='threshold03'
output_folder="/data3/PAMI_Datasets/Replace_Version/VSRD_PP_SPLIT_Ver2/"


python replace_2dbbox_with_intern_image.py     --root_dirname $root_dirname \
                                                --det2d_path $det2d_path \
                                                --threshold_option $threshold_option \
                                                --output_folder $output_folder




}

UPDATE_2D_BBOX_WITH_INTERNIMAGE