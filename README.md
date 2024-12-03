## VSRD and VSRD++ Training the Monocular 3D Detection using MonoFlex.

[Official MonFlex Github Repo](https://github.com/zhangyp15/MonoFlex)


### Training the MonoFlex

change the config file in the `config_train`, choose either `vsrd_configs.py` or `vsrd_pp_configs.py`

Run the code using the following scripts:
```
cd scripts

sh train.sh
```

### Generated the Validation and Evaluation Results `.txt`

```
cd scripts
sh val_evaluation_infernece.sh
```





