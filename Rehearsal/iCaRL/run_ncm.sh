#!/bin/bash

SEED=0
DEVICE='mps'
DEVICE_NAME='macbook'
DATASET='CIFAR100'
MODEL='iCaRL'
EPOCH=1
BATCH_SIZE=512
TEST_SIZE=256
LR=0.5
CL='FC'
INCREMENT=10
MEMORY=2000

export CUDA_VISIBLE_DEVICES=$DEVICE

python -m icarl_ncm --seed $SEED --device $DEVICE --device_name $DEVICE_NAME --dataset $DATASET --batch_size $BATCH_SIZE --test_size $TEST_SIZE --model_name $MODEL --epoch $EPOCH --lr $LR --classifier $CL --class_increment $INCREMENT --memory $MEMORY --fixed_order