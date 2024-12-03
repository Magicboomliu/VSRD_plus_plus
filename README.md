## Stage 2 Training of MonoDETR using VSRD/VSRD++ Pseudo Labels



### Training 

```
bash train.sh configs/monodetr.yaml > logs/monodetr.log
```

### Validation and the Evaluation

```
# train
bash test.sh test_configs/monodetr_validation.yaml


# test
bash test.sh test_configs/monodetr_test.yaml
```