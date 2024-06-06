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
SEED=0
MODEL="slot-attention-clevr-state-$SEED-$NUM"
CUDA_VISIBLE_DEVICES=$DEVICE python clevr_hans/src/supconcepts/train_supconcepts.py --data-dir $DATA \
--seed $SEED --dataset $DATASET --epochs 30 \
--name $MODEL --lr 0.0001 --batch-size 128 --n-slots 4 --n-iters-slot-att 3 --n-attr 18 --train-only \
--fp-pretrained-ckpt "clevr_hans/src/pretrain-slot-attention/runs/slot-attention-clevr-state-$SEED-0/slot-attention-clevr-state-$SEED-0" \

SEED=1
MODEL="slot-attention-clevr-state-$SEED-$NUM"
CUDA_VISIBLE_DEVICES=$DEVICE python clevr_hans/src/supconcepts/train_supconcepts.py --data-dir $DATA \
--seed $SEED --dataset $DATASET --epochs 30 \
--name $MODEL --lr 0.0001 --batch-size 128 --n-slots 4 --n-iters-slot-att 3 --n-attr 18 --train-only \
--fp-pretrained-ckpt "clevr_hans/src/pretrain-slot-attention/runs/slot-attention-clevr-state-$SEED-0/slot-attention-clevr-state-$SEED-0"

SEED=2
MODEL="slot-attention-clevr-state-$SEED-$NUM"
CUDA_VISIBLE_DEVICES=$DEVICE python clevr_hans/src/supconcepts/train_supconcepts.py --data-dir $DATA \
--seed $SEED --dataset $DATASET --epochs 30 \
--name $MODEL --lr 0.0001 --batch-size 128 --n-slots 4 --n-iters-slot-att 3 --n-attr 18 --train-only \
--fp-pretrained-ckpt "clevr_hans/src/pretrain-slot-attention/runs/slot-attention-clevr-state-$SEED-0/slot-attention-clevr-state-$SEED-0"



#CUDA_LAUNCH_BLOCKING=1 python clevr_hans/src/supconcepts/train_supconcepts.py --data-dir $DATA \
#--seed $SEED --dataset $DATASET --epochs 30 \
#--name $MODEL --lr 0.0001 --batch-size 8 --n-slots 4 --n-iters-slot-att 3 --n-attr 18 --train-only \
#--fp-pretrained-ckpt "clevr_hans/src/pretrain-slot-attention/runs/slot-attention-clevr-state-$SEED-0/slot-attention-clevr-state-$SEED-0" \
#--no-cuda
