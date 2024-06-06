#!/bin/bash

# to be called as: python clevr-clot-attention.sh 0 0 /pathtoclevrv1/
# (for cuda device 0 and run 0)

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
SEED=$3
DATA=$4
DATASET=clevr-state
#-------------------------------------------------------------------------------#
# Train on CLEVR_v1
#SEED=0
#MODEL="slot-attention-clevr-state-$SEED-$NUM"
#CUDA_VISIBLE_DEVICES=$DEVICE python train.py --data-dir $DATA --seed $SEED --dataset $DATASET --epochs 500 --name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 18 --train-only
#
#SEED=1
#MODEL="slot-attention-clevr-state-$SEED-$NUM"
#CUDA_VISIBLE_DEVICES=$DEVICE python train.py --data-dir $DATA --seed $SEED --dataset $DATASET --epochs 500 --name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 18 --train-only

SEED=2
MODEL="slot-attention-clevr-state-$SEED-$NUM"
CUDA_VISIBLE_DEVICES=$DEVICE python train.py --data-dir $DATA --seed $SEED --dataset $DATASET --epochs 500 --name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 18 --train-only
