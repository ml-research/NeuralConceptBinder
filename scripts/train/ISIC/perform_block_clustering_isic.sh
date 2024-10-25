#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1

#-------------------------------------------------------------------------------#
CUDA_VISIBLE_DEVICES=$DEVICE python perform_block_clustering_isic.py \
--seed 0 --batch_size 20 --num_workers 0 --image_size 128 --image_channels 3 --log_path logs/ --lr_dvae 3e-4 \
--lr_enc 1e-4 --lr_dec 3e-4 --lr_warmup_steps 15000 --lr_half_life 125000 --clip 0.05 --epochs 1 \
--num_iterations 3 --num_slots 4 --num_blocks 8 --cnn_hidden_size 512 --slot_size 2048 --mlp_hidden_size 192 \
--num_prototypes 64 --vocab_size 4096 --num_decoder_layers 8 --num_decoder_heads 4 --d_model 192 --dropout 0.1 \
--temp 1. --thresh_count_obj_slots -1 --num_clustering_samples 2000 \
--data_path '/workspace/datasets/ISIC19/' \
--checkpoint_path /workspace/repositories/NeuralConceptBinder/logs/isic/isic_seed0/best_model.pt