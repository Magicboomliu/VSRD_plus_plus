# Monocular 3D Detection Using the Pseudo Labels

The VSRD and its extended version VSRD++ are both 2-Stage Pipeline:
- Stage 1: **Multi-View 3D Auto-Labeling** with camera poses and sequential 2D observations including images and its correspoding 2D Masks. 
- Stage 2: **Training of the Monocular 3D Detectors using the Pseo Labels Created in the Stage1**  


In this *README.md*, we tell how to do data processing and training of the Monocular 3D Detectors.


## Step 1: Data Preprocessing (Optional)
For the reason that the pseudo labels created by the VSRD or VSRD++ are recommended to saved as the [KITTI3D Format](https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt), which requires estimated 2D detection box (May be some input of M3D methods, eg, WeakM3D), so we provide a conversion code to generated 2D detection boxes, and updated the pseudo labels as well. 
- Generated Predicted 2D Detection Box using [InternImage](git@github.com:OpenGVLab/InternImage.git). [[Pretrained Models](https://drive.google.com/file/d/10SH7Yrrqis53zzz86SKn0rnBwAyds9GG/view?usp=sharing)]

```
cd preprocessing/segmentation/detection/scripts
sh generated_2d_bounding_box.sh
```
The generated bounding box can be also direcly downloaded [here](https://drive.google.com/file/d/1-ybJmeeQSF-WVm8r0UED53rKjBLEWh6R/view?usp=sharing)

