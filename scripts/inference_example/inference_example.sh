#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
SEED=$2
DATA=$3
CKPT=$4
RETCORPUS=$5

CUDA_VISIBLE_DEVICES=$DEVICE python inference_example.py \
--seed $SEED --image_size 128 --image_channels 3 --clip 0.05 --num_iterations 3 --num_slots 4 --num_blocks 16 \
--cnn_hidden_size 512 --slot_size 2048 --mlp_hidden_size 192 --num_prototypes 64 --vocab_size 4096 \
--num_decoder_layers 8 --num_decoder_heads 4 --d_model 192 --dropout 0.1 \
--data_path $DATA --batch_size 20 --model_type ncb --clf_type dt \
--checkpoint_path $CKPT --seed 0 \
--thresh_count_obj_slot -1 --perc_imgs 1.0 \
--retrieval_encs proto-exem \
--retrieval_corpus_path $RETCORPUS