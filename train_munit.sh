#!/bin/bash

# This file can be used to train models of natural variation for the
# SVHN, GTSRB, and CURE-TSR datasets.

# Variables needed to select dataset and source of natural variation
export DATASET='mnist'
export DATA_DIR=./datasets/mnist
export SOURCE='background-color'
export SZ=32

# Path to MUNIT configuration file.  Edit this file to change the number of iterations, 
# how frequently checkpoints are saved, and other properties of MUNIT.
# The parameter `style_dim` corresponds to the dimension of `delta` in our work.
export CONFIG_PATH=core/models/munit/tiny_munit.yaml

# Output images and checkpoints will be saved to this path.
export OUTPUT_PATH=core/models/munit/results

python3 core/train_munit.py \
    --config $CONFIG_PATH --dataset $DATASET --source-of-nat-var $SOURCE \
    --output_path $OUTPUT_PATH --data-size $SZ --train-data-dir $DATA_DIR
    