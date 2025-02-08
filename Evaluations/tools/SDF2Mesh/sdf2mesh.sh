SDF2MESH(){
root_dirname="/media/zliu/data12/dataset/VSRD_PP_Sync/"
ckpt_dirname="/media/zliu/data12/dataset/TPAMI_Saved_Ckpts/VSRDPP_saved_ckpts/ckpts/"
ckpt_filename="step_2499.pt"
output_folder="/media/zliu/data12/TPAMI_Results/SDF_Visualizations/"



python sdf2mesh.py --root_dirname $root_dirname \
                   --ckpt_dirname $ckpt_dirname \
                   --ckpt_filename $ckpt_filename \
                   --output_folder $output_folder \
                   --use_dynamic_mask \
                   --dynamic_modeling \
                   --use_residual_distance_field
                #    --without_box \


}

SDF2MESH

