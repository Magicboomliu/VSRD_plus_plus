TRAIN_WEAKM3D(){
unset LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 python scripts/train_kitti360.py --config ./config/resnet34_backbone.yaml   

}


TRAIN_WEAKM3D