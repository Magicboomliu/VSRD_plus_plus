TRAIN_MONOFLEX_VSRD24_SPLITS_GT(){
cd ..

batch_size=8
config_path="runs/monoflex.yaml"
output_path="/data1/liu/PAMI_Datasets/Z_Outputs_Models/VSRD24_Splits/GT/"
#'vsrd24_splits_vsrd' , 'vsrd24_splits_vsrdpp' , 'vsrd24_splits_autolabels', 'casual_splits_vsrd', 'casual_splits_vsrdpp', 'casual_splits_autolabels'
split_type='vsrd24_splits_gt' 
pretrained_model_path='none'
code_path="/home/zliu/TPAMI_Revision/PAMI_Revision_Round1/VSRD_plus_plus"

CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --batch_size $batch_size \
                            --config $config_path \
                            --output $output_path \
                            --pretrained_model_path $pretrained_model_path \
                            --split_type $split_type \
                            --code_path $code_path
}

TRAIN_MONOFLEX_Casual_SPLITS_GT(){
cd ..

batch_size=8
config_path="runs/monoflex.yaml"
output_path="/data1/liu/PAMI_Datasets/Z_Outputs_Models/CausalSplit/GT"
#'vsrd24_splits_vsrd' , 'vsrd24_splits_vsrdpp' , 'vsrd24_splits_autolabels', 'casual_splits_vsrd', 'casual_splits_vsrdpp', 'casual_splits_autolabels'
split_type='casual_splits_gt' 
pretrained_model_path='none'
code_path="/home/zliu/TPAMI_Revision/PAMI_Revision_Round1/VSRD_plus_plus"

CUDA_VISIBLE_DEVICES=1 python tools/plain_train_net.py --batch_size $batch_size \
                            --config $config_path \
                            --output $output_path \
                            --pretrained_model_path $pretrained_model_path \
                            --split_type $split_type \
                            --code_path $code_path
}



TRAIN_MONOFLEX_Casual_SPLITS_GT

# TRAIN_MONOFLEX_VSRD24_SPLITS_GT