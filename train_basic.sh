#!/bin/bash

# This script can be used to train classifiers using a pre-trained 
# model of natural variation.

# Select your dataset and source of natural variation:
# Below we list valid choices for sources of natural variation for each dataset
#   * 'mnist': ['background-color']
#   * 'svhn': ['contrast', 'brightness', 'contrast+brightness']
#   * 'gtsrb': ['contrast', 'brightness']
#   * 'cure-tsr': ['snow', 'rain', 'haze', etc] (see https://github.com/olivesgatech/CURE-TSR#challenging-conditions)
export DATASET='mnist'
export TRAIN_DIR=./datasets/mnist/MNIST
export SOURCE='background-color'

# Select your achitecture:
#   * 'basic' is a relatively shallow CNN (see training/classifiers/basic.py)
#   * 'resnet50' was used in our paper for the ImageNet/ImageNet-c experiments
#   * you can also supply any of the models in torchvision.models (https://pytorch.org/docs/stable/torchvision/models.html)
export ARCHITECTURE='basic'
export N_CLASSES=10         # number of classes
export SZ=32                # dataset image size (SZ x SZ x 3)
export BS=256                # batch size

# Select the model of natural variation and the dimension of nuisance space 
# Note that this must match the dimension in models/munit.yaml
export MODEL_PATH=./core/models/learned_models/mnist/mnist-bkgd-color.pt
export CONFIG_PATH=core/models/munit/tiny_munit.yaml
export DELTA_DIM=2

# Other paths for recording data
export LOG_DIR=./logs           # path to log tensorboard files
export SAVE_PATH=./results      # path to save checkpoints and accuracies

# Distributed settings
export N_GPUS_PER_NODE=4
export N_NODES=1

ulimit -n 4096
python3 -m torch.distributed.launch \
    --nproc_per_node=$N_GPUS_PER_NODE --nnodes=$N_NODES --node_rank=0 core/train.py \
    --train-data-dir $TRAIN_DIR --logdir $LOG_DIR --save-path $SAVE_PATH \
    --data-size $SZ --batch-size $BS --num-classes $N_CLASSES --init-bn0 --no-bn-wd --half-prec \
    --architecture $ARCHITECTURE --delta-dim $DELTA_DIM --config $CONFIG_PATH \
    --dataset $DATASET --model-paths $MODEL_PATH --source-of-nat-var $SOURCE \
    --distributed --setup-verbose --optimizer adadelta \
    --phases "[{'ep': 0}, {'ep': (0, 100), 'lr': (1.0, 1.0)}]"

