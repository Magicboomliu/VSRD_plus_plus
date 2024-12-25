TRAIN_MONOFLEX(){
cd ..

batch_size=8
config_path="runs/monoflex.yaml"
output_path="/data1/KITTI360/Z_Outputs_Models/VSRD24_Splits/Autolabels"
#'vsrd24_splits_vsrd' , 'vsrd24_splits_vsrdpp' , 'vsrd24_splits_autolabels', 'casual_splits_vsrd', 'casual_splits_vsrdpp', 'casual_splits_autolabels'
split_type='vsrd24_splits_autolabels' 
pretrained_model_path='/home/zliu/TPAMI25/Mono3Ds/model_checkpoint.pth'
code_path="/home/zliu/TPAMI25/Mono3Ds/MonoFlex"

CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --batch_size $batch_size \
                            --config $config_path \
                            --output $output_path \
                            --pretrained_model_path $pretrained_model_path \
                            --split_type $split_type \
                            --code_path $code_path
}

TRAIN_MONOFLEX