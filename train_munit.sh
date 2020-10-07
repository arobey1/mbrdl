#!/bin/bash


export DATASET='gtsrb'
export SOURCE='brightness'
export SZ=64

# Path to MUNIT configuration file
# You can edit this file to change the number of iterations, how frequently
# checkpoints are saved, and other properties of MUNIT.
# The parameter `style_dim` corresponds to the dimension of `delta` in our work.
export CONFIG_PATH=core/models/munit/munit.yaml

# Output images and checkpoints will be saved to this path.
export OUTPUT_PATH=core/models/munit/results

python3 core/train_munit.py \
    --config $CONFIG_PATH --dataset $DATASET --source-of-nat-var $SOURCE \
    --output_path $OUTPUT_PATH --data-size $SZ