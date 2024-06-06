#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
DATA=$2
CKPT=$3
NCATS=$4
NBLOCKS=$5

#-------------------------------------------------------------------------------#
CUDA_VISIBLE_DEVICES=$DEVICE python perform_block_clustering.py \
--seed 0 --batch_size 16 --num_workers 0 --image_size 128 --image_channels 3 --log_path logs/ --lr_dvae 3e-4 \
--lr_enc 1e-4 --lr_dec 3e-4 --lr_warmup_steps 1 --lr_half_life 250000 --clip 0.05 --epochs 1 \
--num_iterations 3 --num_slots 4 --num_blocks $NBLOCKS --cnn_hidden_size 512 --slot_size 2048 --mlp_hidden_size 192 \
--num_prototypes 64 --vocab_size 4096 --num_decoder_layers 8 --num_decoder_heads 4 --d_model 192 --dropout 0.1 \
--temp 1. --num_categories $NCATS \
--data_path $DATA --checkpoint_path $CKPT
