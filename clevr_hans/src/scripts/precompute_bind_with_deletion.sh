#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
SEED=$2
DATA=$3

#-------------------------------------------------------------------------------#
SEED=0
CUDA_VISIBLE_DEVICES=$DEVICE python clevr_hans/src/get_encs_clevr_hans.py \
--data-dir $DATA --num-workers 0 --batch-size 128 --thresh_count_obj_slots 0 --num_blocks 16 \
--retrieval_corpus_path "logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/block_concept_dicts.pkl" \
--checkpoint_path "logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/best_model.pt" \
--feedback_path "clevr_hans/src/runs/CLEVR_4_1_cls/conf_ncb_settransformer-1obj-0-CLEVR_4_1_cls_seed$SEED/neg_feedback_dict.pkl"

SEED=1
CUDA_VISIBLE_DEVICES=$DEVICE python clevr_hans/src/get_encs_clevr_hans.py \
--data-dir $DATA --num-workers 0 --batch-size 128 --thresh_count_obj_slots 0 --num_blocks 16 \
--retrieval_corpus_path "logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/block_concept_dicts.pkl" \
--checkpoint_path "logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/best_model.pt" \
--feedback_path "clevr_hans/src/runs/CLEVR_4_1_cls/conf_ncb_settransformer-1obj-0-CLEVR_4_1_cls_seed$SEED/neg_feedback_dict.pkl"

SEED=2
CUDA_VISIBLE_DEVICES=$DEVICE python clevr_hans/src/get_encs_clevr_hans.py \
--data-dir $DATA --num-workers 0 --batch-size 128 --thresh_count_obj_slots 0 --num_blocks 16 \
--retrieval_corpus_path "logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/block_concept_dicts.pkl" \
--checkpoint_path "logs/clevr4_600_epochs/clevr4_sysbind_orig_seed$SEED/best_model.pt" \
--feedback_path "clevr_hans/src/runs/CLEVR_4_1_cls/conf_ncb_settransformer-1obj-0-CLEVR_4_1_cls_seed$SEED/neg_feedback_dict.pkl"
