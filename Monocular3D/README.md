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
The generated bounding box can be also direcly downloaded [here](https://drive.google.com/file/d/1-ybJmeeQSF-WVm8r0UED53rKjBLEWh6R/view?usp=sharing). 


## Step 2: Symbol-Link the psuedo labels into KITTI3D Format. 

For the reason that the generated pseudo labels with the same name of the KITTI360 filename, we **re-organized** the structure the pseduo labels with the same structure and the filename.   

we provide two kinds of methods to split the training and the validation sets.
- CVPR 2024 Split (Splited by the Sequential Name)
- PAMI 2025 Split (Splited by the Dynamic/Static)


- Re-organize the Pseudo Labels.

```
cd Monocular3D/preprocessing/scripts
# switch the scripts into sequences-based (VSRD) splits  #`SPLIT_THE_FNAME_BY_SEQUENCE` 

sh pre_dataset_fname_split.sh

# `SPLIT_THE_FNAME_BY_DYNAMIC`

sh pre_dataset_fname_split.sh
```

- Replace the Pseudo Labels 2D Detections with Intern-Image
```
cd Monocular3D/preprocessing/scripts

sh post_dataset_fname_processing.sh
```

### Attention !

Note that the above code for generated the static/dynamic splits contains some `random` inside it. In order to reproduce the results, please using the following `sync_relationship` and the `ImageSets`.

- [Sync Relationship Files & ImageSets Files](https://drive.google.com/file/d/1FJBaB4ATNj3D4yICc_XpohKQU_WcpNGy/view?usp=sharing) 

```
cd Monocular3D/preprocessing/scripts

# `SPLIT_THE_FNAME_BY_DYNAMIC_From_SYNC_FILE`
sh pre_dataset_fname_split.sh

```


## Training of the Monocular Depth Estimators
Please refer to different branchs to train the monocular depth estimators.

- MonoFlex
- MonoDeTR
- WeakM3D