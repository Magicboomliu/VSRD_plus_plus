
InternImage_Inference(){

cd ..

datapath="/data3/VSRD_PP_Sync/"
threshold=0.3
filename_list="filenames/KITTI360_DataAll.txt"
config="/home/zliu/TPAMI25/VSRD_plus_plus/preprocessing/segmentation/detection/configs/coco/cascade_internimage_xl_fpn_3x_coco.py"
checkpoint="/home/zliu/TPAMI25/Pretrained_Models/InternImage/cascade_internimage_xl_fpn_3x_coco.pth"

python detection_generation.py --datapath $datapath \
                    --threshold $threshold \
                    --filename_list $filename_list \
                    --config $config \
                    --checkpoint $checkpoint


}

InternImage_Inference