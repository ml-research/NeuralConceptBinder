#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
SEED=$2
DATA=$3
NCATS=$4
PIMGS=$5
#RETCORPUS=$6
#CKPT=$3
#MODELTYPE=$4

#MODELBASE=logs/clevr_easy_500_epochs

# Note the finetuned models represent models that were trained for 500 epochs on the original dataset, we then took
# the best model from those runs and finetuned for 2 epochs on the training set of CLEVR-Easy-1 and are now taking the
# last model from that finetuning

printf "\n#-------------------------------------------------------------------------------#\n"
printf "#-------------------------------------------------------------------------------#\n"
printf "sysbind orig\n"
CUDA_VISIBLE_DEVICES=$DEVICE python analysis_via_clf.py \
--data_path $DATA --batch_size 20 --model_type sysbind --clf_type dt \
--checkpoint_path logs/clevr_easy_500_epochs/sysbind_orig_seed${SEED}_finetune/last_model.pt --seed 0 \
--thresh_count_obj_slot 0 --num_categories $NCATS --perc_imgs $PIMGS 

printf "#-------------------------------------------------------------------------------#\n"
printf "sysbind orig (hard)\n"
CUDA_VISIBLE_DEVICES=$DEVICE python analysis_via_clf.py \
--data_path $DATA --batch_size 20 --model_type sysbind --clf_type dt \
--checkpoint_path logs/clevr_easy_500_epochs/sysbind_orig_seed${SEED}_finetune/last_model.pt --seed 0 \
--thresh_count_obj_slot 0 --num_categories $NCATS --perc_imgs $PIMGS \
--binarize

printf "\n#-------------------------------------------------------------------------------#\n"
printf "sysbind orig (attn)\n"
CUDA_VISIBLE_DEVICES=$DEVICE python analysis_via_clf.py \
--data_path $DATA --batch_size 20 --model_type sysbind --clf_type dt \
--checkpoint_path logs/clevr_easy_500_epochs/sysbind_orig_seed${SEED}_finetune/last_model.pt --seed 0 \
--thresh_count_obj_slot 0 --num_categories $NCATS --perc_imgs $PIMGS \
--binarize --attention_codes

printf "\n#-------------------------------------------------------------------------------#\n"
printf "#-------------------------------------------------------------------------------#\n"
printf "sysbind hard\n"
CUDA_VISIBLE_DEVICES=$DEVICE python analysis_via_clf.py \
--data_path $DATA --batch_size 20 --model_type sysbind_hard --clf_type dt \
--checkpoint_path logs/clevr_easy_500_epochs/sysbind_hard_seed${SEED}_finetune/last_model.pt --seed 0 \
--thresh_count_obj_slot 0 --num_categories $NCATS --perc_imgs $PIMGS --binarize

printf "\n#-------------------------------------------------------------------------------#\n"
printf "sysbind hard (attn)\n"
CUDA_VISIBLE_DEVICES=$DEVICE python analysis_via_clf.py \
--data_path $DATA --batch_size 20 --model_type sysbind_hard --clf_type dt \
--checkpoint_path logs/clevr_easy_500_epochs/sysbind_hard_seed${SEED}_finetune/last_model.pt --seed 0 \
--thresh_count_obj_slot 0 --num_categories $NCATS --perc_imgs $PIMGS --binarize \
--attention_codes

printf "\n#-------------------------------------------------------------------------------#\n"
printf "sysbind step\n"
CUDA_VISIBLE_DEVICES=$DEVICE python analysis_via_clf.py \
--data_path $DATA --batch_size 20 --model_type sysbind_step --clf_type dt \
--checkpoint_path logs/clevr_easy_500_epochs/sysbind_step_seed${SEED}_finetune/last_model.pt --seed 0 \
--thresh_count_obj_slot 0 --num_categories $NCATS --perc_imgs $PIMGS --binarize

printf "\n#-------------------------------------------------------------------------------#\n"
printf "sysbind step (attn)\n"
CUDA_VISIBLE_DEVICES=$DEVICE python analysis_via_clf.py \
--data_path $DATA --batch_size 20 --model_type sysbind_step --clf_type dt \
--checkpoint_path logs/clevr_easy_500_epochs/sysbind_step_seed${SEED}_finetune/last_model.pt --seed 0 \
--thresh_count_obj_slot 0 --num_categories $NCATS --perc_imgs $PIMGS --binarize \
--attention_codes

printf "\n#-------------------------------------------------------------------------------#\n"
printf "#-------------------------------------------------------------------------------#\n"
printf "ncb just prototypes\n"
CUDA_VISIBLE_DEVICES=$DEVICE python analysis_via_clf.py \
--data_path $DATA --batch_size 20 --model_type ncb --clf_type dt \
--checkpoint_path logs/clevr_easy_500_epochs/sysbind_orig_seed${SEED}/best_model.pt --seed 0 \
--thresh_count_obj_slot 0 --num_categories $NCATS --perc_imgs $PIMGS \
--retrieval_encs proto \
--retrieval_corpus_path logs/clevr_easy_500_epochs/sysbind_orig_seed${SEED}/block_concept_dicts.pkl

printf "\n#-------------------------------------------------------------------------------#\n"
printf "ncb just prototypes & exemplars\n"
CUDA_VISIBLE_DEVICES=$DEVICE python analysis_via_clf.py \
--data_path $DATA --batch_size 20 --model_type ncb --clf_type dt \
--checkpoint_path logs/clevr_easy_500_epochs/sysbind_orig_seed${SEED}/best_model.pt --seed 0 \
--thresh_count_obj_slot 0 --num_categories $NCATS --perc_imgs $PIMGS \
--retrieval_encs proto-exem --topk 5 --majority_vote \
--retrieval_corpus_path logs/clevr_easy_500_epochs/sysbind_orig_seed${SEED}/block_concept_dicts.pkl

printf "\n#-------------------------------------------------------------------------------#\n"
printf "ncb prototypes & exemplars argmax selection\n"
CUDA_VISIBLE_DEVICES=$DEVICE python analysis_via_clf.py \
--data_path $DATA --batch_size 20 --model_type ncb --clf_type dt \
--checkpoint_path logs/clevr_easy_500_epochs/sysbind_orig_seed${SEED}/best_model.pt --seed 0 \
--thresh_count_obj_slot 0 --num_categories $NCATS --perc_imgs $PIMGS \
--retrieval_encs proto-exem \
--retrieval_corpus_path logs/clevr_easy_500_epochs/sysbind_orig_seed${SEED}/block_concept_dicts.pkl

printf "\n#-------------------------------------------------------------------------------#\n"
printf "ncb prototypes & exemplars & basis vectors\n"
CUDA_VISIBLE_DEVICES=$DEVICE python analysis_via_clf.py \
--data_path $DATA --batch_size 20 --model_type ncb --clf_type dt \
--checkpoint_path logs/clevr_easy_500_epochs/sysbind_orig_seed${SEED}/best_model.pt --seed 0 \
--thresh_count_obj_slot 0 --num_categories $NCATS --perc_imgs $PIMGS \
--retrieval_encs proto-exem-basis --topk 5 --majority_vote \
--retrieval_corpus_path logs/clevr_easy_500_epochs/sysbind_orig_seed${SEED}/block_concept_dicts.pkl

printf "\n#-------------------------------------------------------------------------------#\n"
printf "ncb prototypes & exemplars & basis vectors argmax selection\n"
CUDA_VISIBLE_DEVICES=$DEVICE python analysis_via_clf.py \
--data_path $DATA --batch_size 20 --model_type ncb --clf_type dt \
--checkpoint_path logs/clevr_easy_500_epochs/sysbind_orig_seed${SEED}/best_model.pt --seed 0 \
--thresh_count_obj_slot 0 --num_categories $NCATS --perc_imgs $PIMGS \
--retrieval_encs proto-exem-basis \
--retrieval_corpus_path logs/clevr_easy_500_epochs/sysbind_orig_seed${SEED}/block_concept_dicts.pkl

printf "\n#-------------------------------------------------------------------------------#\n"
printf "nlotm\n"
CUDA_VISIBLE_DEVICES=$DEVICE python analysis_via_clf.py \
--image_size 128 --image_channels 3 --clip 0.05 --num_iterations 3 --num_slots 4 --num_blocks 8 \
--num_categories $NCATS --perc_imgs $PIMGS \
--data_path $DATA --batch_size 20 --model_type nlotm --clf_type dt \
--checkpoint_path logs/nlotm/clevr_easy/checkpoint_seed_${SEED}.pt.tar --seed 0 
