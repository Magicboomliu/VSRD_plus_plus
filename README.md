# VSRD++: Improving Weakly Supervised 3D Detection
Volumetric Silhouette Rendering for Dynamic Objects


## Installation

1. Setup the conda environment.

```bash
conda env create -f environment.yaml
```

2. Install this repository.

```bash
pip install -e .
```  


## Data Preparation

1. Download the Custom `KITTI360` dataset from [google drive](https://drive.google.com/file/d/1syBPCdU0Hs2AWgQfsPohqMFXNEpM3eWV/view?usp=sharing).  Or you can download with the `curl` command.

```
curl -H "Authorization: Bearer ya29.a0AcM612x77zhxvd1UVcnK8cd_YXFeMKJbZXZX0zllYv52vBfHMIdhICj7fWe_-M1mAMa-1pfNFtlZXZOJ7gb-LsK246eOE8sDVUOdpOR4eH0dOTDJ6k9yoLRr07u7tQt18jb7PSUshEWJqqY0aLdxTJSB3dp2FwA3ZpEOP-JyaCgYKAakSARESFQHGX2MixGEaSso_fx9nhTaCDF8UDA0175" https://www.googleapis.com/drive/v3/files/1syBPCdU0Hs2AWgQfsPohqMFXNEpM3eWV?alt=media -o KITTI360_For_Upload.zip
```

Then `unzip` the dataset and create the soft-link for training.  
- Unzip files
```
unzip KITTI360_For_Upload.zip

find /path/to/directory -type l -exec rm -v {} +
```
- Create the soft-link for training

```
cd preprocessing/data_organization
python soft_link.py --soft_linked_folder $your_target_path --source_root_folder $your_source_path
```
- Update the Dynamic Filenames and the Sample Filenames
```
python update_sampled_image_filenames.py --gt_original_root_folder $your_root_path --changed_root_folder_root $your_changed_root_path  

python update_configs.py  --root_path "/data1/liu/VSRD_PP_Sync" --configs_path "/home/zliu/CVPR2025/VSRD-V2/Optimized_Based/configs
```

## RUN THE TRAINING OF VSRD++ Dynamic Auto-Labeling.
We can find the following code in the `Optimized_Based` folder


```
cd  Optimized_Based/scripts
sh DDP_RUN.sh
```

where inside the `DDP_RUN.sh` comes the following command for DDP training, one using `Attribute Initialization` using Self-supervised IGEVStereo(Sceneflow pretrained), named`TRAIN_DDP_VSRDPP`; the other is without using the `Attribute Initialization`, named `TRAIN_DDP_VSRD_SIMPLE`
```
# using the initialization
TRAIN_DDP_VSRDPP(){
cd ..
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29500 \
    --nnodes 1 \
    --nproc_per_node 2 \
    train_sequence_ddp.py --config_path "00" \
    --device_id 0

}

# without using the initialization
TRAIN_DDP_VSRD_SIMPLE(){
cd ..
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29501 \
    --nnodes 1 \
    --nproc_per_node 2 \
    train_without_initial_sequence_ddp.py --config_path "test" \
    --device_id 0


}

TRAIN_DDP_VSRD_SIMPLE
# TRAIN_DDP_VSRDPP
```


## Ablaion Studies Settings  
Here we provide 2 main ablation studies settings, 
- using the `Dynamic Modeling` or not. 
- Using the `Psuedo Attribute Initialization` or not.

We can change the setting in each seuqnce's config file in `Optimized_Based/configs`, with the following lines:  


```
_C.TRAIN.USE_RDF_MODELING=True
_C.TRAIN.USE_DYNAMIC_MASK=True
_C.TRAIN.USE_DYNAMIC_MODELING=True

# Just for testing
_C.TRAIN.DYNAMIC_LABELS_PATH="/data1/liu/VSRD_PP_Sync/est_dynamic_list/sync00/dynamic_mask.txt"

# selective from 'mlp', 'vector_velocity','scalar_velocity'
_C.TRAIN.DYNAMIC_MODELING_TYPE='vector_velocity'

```


## RELATED SUB-README 
- [Optimization-Based Scene-Wise  Multi-view 3D Bounding Box Rendering.](Optimized_Based/)  


- [Dynamic and Static Instance Classification using 2D Flow and Depth Consistency  
](preprocessing/)

```cd  preprocessing/```


