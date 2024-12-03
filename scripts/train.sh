TRAIN_MONOFLEX(){
cd ..

batch_size=8
config_path="runs/monoflex.yaml"
output_path="output/exp_staic_dynamic_split_vsrd_pp"
split_type='vsrd_pp'
pretrained_model_path='none'
code_path="/home/zliu/TPAMI25/Mono3Ds/MonFlex"

CUDA_VISIBLE_DEVICES=3 python tools/plain_train_net.py --batch_size $batch_size \
                            --config $config_path \
                            --output $output_path \
                            --pretrained_model_path $pretrained_model_path \
                            --split_type $split_type \
                            --code_path $code_path
}

TRAIN_MONOFLEX