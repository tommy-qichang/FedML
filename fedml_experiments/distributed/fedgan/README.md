## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments

## BRATS generation experiments
3 clients training, T2 data, 50 rounds, 2 epochs per round, batch size 10:
```
sh run_fedgan_distributed_pytorch.sh 3 3 dadgan hetero 50 2 10 sgd 0.01 brats_t2 "./../../../data/brats" mapping_config_sense02_3 dadgan
```

##run on background
```
nohup sh run_fedgan_distributed_pytorch.sh 3 3 dadgan hetero 50 2 10 sgd 0.01 brats_t2 "./../../../data/brats" mapping_config_sense02_3 dadgan > ./fedgan-brats-r50.txt 2>&1 &
```
