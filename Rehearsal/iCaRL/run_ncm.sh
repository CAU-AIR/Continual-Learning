#!/bin/bash

SEED=0
DEVICE=0
DEVICE_NAME='hspark'
DATASET='CIFAR10'
MODEL='iCaRL'
EPOCH=30
BATCH_SIZE=512
TEST_SIZE=256
CL='NCM'
INCREMENT=1

export CUDA_VISIBLE_DEVICES=$DEVICE

python3 -m main --seed $SEED --device $DEVICE --device_name $DEVICE_NAME --dataset $DATASET --batch_size $BATCH_SIZE --test_size $TEST_SIZE --model_name $MODEL --epoch $EPOCH --classifier $CL --class_increment $INCREMENT