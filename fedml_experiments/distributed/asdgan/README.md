## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments

## AsynDGAN on BRATS experiments
3 clients training, T2 data, 200 epochs, batch size 20/client (sample method 'balance'):
```
sh run_asdgan_distributed_pytorch.sh 3 3 asdgan hetero 200 20 adam 0.001 balance brats_t2 "./../../../data/brats" mapping_config_sense02_3 asdgan
```

```
sh run_asdgan_distributed_pytorch.sh 3 3 asdgan hetero 200 20 sgd 0.001 balance brats_t2 "/research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2" mapping_config_santorini_cs_3 asdgan
```
## run on background
```
nohup sh run_asdgan_distributed_pytorch.sh 3 3 asdgan hetero 200 20 adam 0.001 balance brats_t2 "./../../../data/brats" mapping_config_sense02_3 asdgan > ./asdgan-brats-r200.txt 2>&1 &
```

## save synthetic images
save 3 * 20 synthetic images to visualize
```
python save_syn.py --dataset brats_t2 --batch_size 20 --save_dir ./run/brats_t2/asdgan/experiment_0 --load_filename G_aggregated_checkpoint_ep50.pth.tar --epoch 50 --GPUid 0 --num_test 3
```