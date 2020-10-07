#!/bin/bash

# This script can be used to train classifiers on ImageNet/ImageNet-c 
# using a pre-trained model of natural variation.

# Select the ImageNet dataset and proivde paths to train/val sets
export DATASET='imagenet'
export TRAIN_DIR=./datasets/imagenet/train
export VAL_DIR=./datasets/imagenet-c/weather/snow/3

# Select your achitecture:
#   * 'resnet50' was used in our paper for the ImageNet/ImageNet-c experiments
#   * you can also supply any of the models in torchvision.models (https://pytorch.org/docs/stable/torchvision/models.html)
export ARCHITECTURE='resnet50'
export N_CLASSES=50           # number of classes
export SZ=224                   # dataset image size (SZ x SZ x 3)
export BS=64                    # batch size

# Select the model of natural variation and the dimension of nuisance space 
# Note that this must match the dimension in models/munit.yaml
export MODEL_PATH=./core/models/learned_models/imagenet/imagenet-snow-1.pt
export DELTA_DIM=8

# Other paths for recording data
export LOG_DIR=./logs           # path to log tensorboard files
export SAVE_PATH=./results      # path to save checkpoints and accuracies

# Distributed settings
export N_GPUS_PER_NODE=4
export N_NODES=1

ulimit -n 4096
python -m torch.distributed.launch \
    --nproc_per_node=$N_GPUS_PER_NODE --nnodes=$N_NODES --node_rank=0 core/train.py \
    --train-data-dir $TRAIN_DIR --val-data-dir $VAL_DIR --logdir $LOG_DIR \
    --save-path $SAVE_PATH --init-bn0 --no-bn-wd --half-prec \
    --data-size $SZ --batch-size $BS --num-classes $N_CLASSES \
    --architecture $ARCHITECTURE --delta-dim $DELTA_DIM  \
    --dataset $DATASET --model-paths $MODEL_PATH \
    --distributed --setup-verbose --optimizer sgd \
    --phases "[{'ep': 0}, {'ep': (0, 200), 'lr': (0.05, 0.001)}]"


