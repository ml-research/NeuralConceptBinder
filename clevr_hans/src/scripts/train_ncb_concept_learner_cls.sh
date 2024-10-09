#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
SEED=$3
DATA=$4
# e.g., DATA=/workspace/datasets/CLEVR_4_1_cls/

#-------------------------------------------------------------------------------#
# CLEVR-Hans3

SEED=0
CUDA_VISIBLE_DEVICES=$DEVICE python clevr_hans/src/train_ncb_nesy_concept_learner.py \
--data-dir $DATA --epochs 30 --num-workers 0 \
--name "ncb_settransformer-1obj-$NUM" --lr 0.0001 --batch-size 128 --seed $SEED --mode train \
--retrieval_corpus_path "/workspace/repositories/SysBindRetrieve/logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/block_concept_dicts.pkl" \
--checkpoint_path "/workspace/repositories/SysBindRetrieve/logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/best_model.pt" \
--precompute-bind

SEED=1
CUDA_VISIBLE_DEVICES=$DEVICE python clevr_hans/src/train_ncb_nesy_concept_learner.py \
--data-dir $DATA --epochs 30 --num-workers 0 \
--name "ncb_settransformer-1obj-$NUM" --lr 0.0001 --batch-size 512 --seed $SEED --mode train \
--retrieval_corpus_path "/workspace/repositories/SysBindRetrieve/logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/block_concept_dicts.pkl" \
--checkpoint_path "/workspace/repositories/SysBindRetrieve/logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/best_model.pt" \
--precompute-bind

SEED=2
CUDA_VISIBLE_DEVICES=$DEVICE python clevr_hans/src/train_ncb_nesy_concept_learner.py \
--data-dir $DATA --epochs 30 --num-workers 0 \
--name "ncb_settransformer-1obj-$NUM" --lr 0.0001 --batch-size 512 --seed $SEED --mode train \
--retrieval_corpus_path "/workspace/repositories/SysBindRetrieve/logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/block_concept_dicts.pkl" \
--checkpoint_path "/workspace/repositories/SysBindRetrieve/logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/best_model.pt" \
--precompute-bind
