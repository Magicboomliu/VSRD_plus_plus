# Optimization-Based Scene-Wise  Multi-view 3D Bounding Box Rendering.

Following the Vanila VSRD, we propsoed the VSRD++ consider the dynamic object modeling. We using the following three method for dynamic modeling: 

- Instance Residual Field (via MLP)
- Vector Velocity Modeling 
- Scalar Velocity Modeling 

[Detailed Slide Slide](https://docs.google.com/presentation/d/1B2l-yRS63q4lu8Qb-4qCMdWHLMMMLU5L/edit?usp=sharing&ouid=112605403951022205460&rtpof=true&sd=true)

![image](figures/mlp.png)
![image](figures/velocity.png)


## Training 

````
python train.py

````

## Inference

```
python infernece.py

```

## evaluations

```
python evaluation.py
```

Change the dynamic modeling here at the `configs` here

```
# Modifications Here
_C.TRAIN.MODEL_TYPE='box_residual_scalar_velocity'
_C.TRAIN.OPTIMIZATION_NUM_STEPS=4000
_C.TRAIN.OPTIMIZATION_WARMUP_STEPS=1000
_C.TRAIN.OPTIMIZATION_RESIDUAL_BOX_STEPS=2000

_C.TRAIN.USE_RDF_MODELING=True
_C.TRAIN.USE_DYNAMIC_MASK=True
_C.TRAIN.USE_DYNAMIC_MODELING=True


# Just for testing
_C.TRAIN.DYNAMIC_LABELS_PATH="/home/zliu/Desktop/CVPR2025/VSRD-V2/data_pre_processing/Dyanmic_Object_Filtering/saved_contents/sync06/dynamic_mask.txt"

# selective from 'mlp', 'vector_velocity','scalar_velocity'
_C.TRAIN.DYNAMIC_MODELING_TYPE='scalar_velocity'
``