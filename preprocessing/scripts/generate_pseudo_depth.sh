Depth_Generation_Using_LEAStereo(){
cd ..
cd disparity_estimation/Leastereo/
root_folder="/media/zliu/data12/dataset/KITTI/VSRD_Format/data_2d_raw/"
saved_name='pseudo_leastereo_depth'

CUDA_VISIBLE_DEVICES=0 python sequential_depth_estimation.py \
                --kitti2015=1    --maxdisp=192 \
                --crop_height=384  --crop_width=1416  \
                --data_path='None' \
                --test_list="none"\
                --save_path='./predict_depth/' \
                --fea_num_layer 6 --mat_num_layers 12\
                --fea_filter_multiplier 8 --fea_block_multiplier 4 --fea_step 3  \
                --mat_filter_multiplier 8 --mat_block_multiplier 4 --mat_step 3  \
                --net_arch_fea='run/sceneflow/best/architecture/feature_network_path.npy' \
                --cell_arch_fea='run/sceneflow/best/architecture/feature_genotype.npy' \
                --net_arch_mat='run/sceneflow/best/architecture/matching_network_path.npy' \
                --cell_arch_mat='run/sceneflow/best/architecture/matching_genotype.npy' \
                --resume './run/Kitti15/best/best.pth' \
                --root_folder $root_folder \
                --saved_name $saved_name 

}


Depth_Generation_Using_LEAStereo