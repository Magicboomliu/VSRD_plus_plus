
InternImage_Inference(){

image=/home/liuzihua/InternImage/detection/example2.png
config=/home/zliu/VSRD_plus_plus/preprocessing/segmentation/detection/configs/coco/cascade_internimage_xl_fpn_3x_coco.py
checkpoint=/home/zliu/pretrained_models/cascade_internimage_xl_fpn_3x_coco.pth


python detection_generation.py --img $image \
                    --config $config \
                    --checkpoint $checkpoint


}

InternImage_Inference