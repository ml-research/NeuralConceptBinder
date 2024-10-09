#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
SEED=$3
#DATA=$4


#-------------------------------------------------------------------------------#
# CLEVR-Hans3

CUDA_VISIBLE_DEVICES=$DEVICE python clevr_hans/src/train_nesy_concept_learner.py \
--data-dir /workspace/datasets_wolf/CLEVR_4_1_cls/ --num-workers 0 \
--name "ncb_settransformer-1obj-$NUM" --batch-size 512 --seed $SEED --mode expls --expl_thresh 0.25 \
--retrieval_corpus_path "/workspace/repositories/SysBindRetrieve/logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/block_concept_dicts.pkl" \
--checkpoint_path "/workspace/repositories/SysBindRetrieve/logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/best_model.pt" \
--precompute-bind \
--fp-ckpt "/workspace/repositories/SysBindRetrieve/clevr_hans/src/runs/CLEVR_4_1_cls/ncb_settransformer-1obj-$NUM-CLEVR_4_1_cls_seed$SEED/*_bestvalloss_*.pth"