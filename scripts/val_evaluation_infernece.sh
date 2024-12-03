VAL_INFERAENCE(){
cd ..

batch_size=8
config_path="runs/monoflex_val.yaml"
output_path="output/Inference/Static_Dyanmic/VAL/VSRDPP"
evaluation_type='val'
pretrained_model_path='/home/zliu/TPAMI25/SensetimeJapan_Internship/model_moderate_best_soft.pth'
code_path="/home/zliu/TPAMI25/Mono3Ds/MonFlex"

CUDA_VISIBLE_DEVICES=0 python tools/test_validation_inference.py --batch_size $batch_size \
                            --config $config_path \
                            --output $output_path \
                            --pretrained_model_path $pretrained_model_path \
                            --evaluation_type $evaluation_type \
                            --code_path $code_path

}


TEST_INFERENCE(){
cd ..

batch_size=8
config_path="runs/monoflex_test.yaml"
output_path="output/Inference/Static_Dyanmic/Test/VSRDPP"
evaluation_type='test'
pretrained_model_path='/home/zliu/TPAMI25/SensetimeJapan_Internship/model_moderate_best_soft.pth'
code_path="/home/zliu/TPAMI25/Mono3Ds/MonFlex"

CUDA_VISIBLE_DEVICES=0 python tools/test_validation_inference.py --batch_size $batch_size \
                            --config $config_path \
                            --output $output_path \
                            --pretrained_model_path $pretrained_model_path \
                            --evaluation_type $evaluation_type \
                            --code_path $code_path


}


# VAL_INFERAENCE
TEST_INFERENCE