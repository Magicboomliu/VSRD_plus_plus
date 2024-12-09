## 3D Auto-Labeling For Traning on Tsubame

### Step 1: Generate the sub training scripts

change `output_dir` to a specific location, also change the output
` --saved_ckpt_path \"/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/output_models_ablations\""`

```
cd Optimized_Based/scripts
sh generate_sub_trainer_scripts.sh
```
