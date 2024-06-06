#!/bin/bash

blender=/usr/bin/blender-2.78c-linux-glibc219-x86_64/blender

#----------------------------------------------------------#
NUM_TRAIN_SAMPLES=5000
NUM_VAL_SAMPLES=2000
NUM_TEST_SAMPLES=2000
NUM_SUDOKU_SAMPLES=10000

NUM_PARALLEL_THREADS=30
NUM_THREADS=6
MIN_OBJECTS=1
MAX_OBJECTS=1
MAX_RETRIES=30

FILENAME_PREFIX=CLEVR_4
##----------------------------------------------------------#
#
## generate training images
#for CLASS_ID in 0
#do
#time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/clevr_4_1/train/images/ --output_scene_dir ../output/clevr_4_1/train/scenes/ --output_scene_file ../output/clevr_4_1/train/Clevr_4_scenes_train.json --filename_prefix $FILENAME_PREFIX --max_retries $MAX_RETRIES --num_images $NUM_TRAIN_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 128 --height 128 --properties_json data/properties_Clevr_4.json --conf_class_combos_json data/Clevr_4_GTClasses.json --gt_class_combos_json data/Clevr_4_GTClasses.json --img_class_id $CLASS_ID
#done
#
## merge all classes join files to one json file
#python merge_json_files.py --json_dir ../output/clevr_4_1/train/ --filename_prefix $FILENAME_PREFIX
#
###----------------------------------------------------------#
#
## generate test images
#for CLASS_ID in 0
#do
#time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/clevr_4_1/test/images/ --output_scene_dir ../output/clevr_4_1/test/scenes/ --output_scene_file ../output/clevr_4_1/test/Clevr_4_scenes_test.json --filename_prefix $FILENAME_PREFIX --max_retries $MAX_RETRIES --num_images $NUM_TEST_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 128 --height 128 --properties_json data/properties_Clevr_4.json --conf_class_combos_json data/Clevr_4_GTClasses.json --gt_class_combos_json data/Clevr_4_GTClasses.json --img_class_id $CLASS_ID
#done
#
#
## merge all classes join files to one json file
#python merge_json_files.py --json_dir ../output/clevr_4_1/test/ --filename_prefix $FILENAME_PREFIX
#
###----------------------------------------------------------#
#
## generate val images
#for CLASS_ID in 0
#do
#time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/clevr_4_1/val/images/ --output_scene_dir ../output/clevr_4_1/val/scenes/ --output_scene_file ../output/clevr_4_1/val/Clevr_4_scenes_val.json --filename_prefix $FILENAME_PREFIX --max_retries $MAX_RETRIES --num_images $NUM_VAL_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 128 --height 128 --properties_json data/properties_Clevr_4.json --conf_class_combos_json data/Clevr_4_GTClasses.json --gt_class_combos_json data/Clevr_4_GTClasses.json --img_class_id $CLASS_ID
#done
#
## merge all classes join files to one json file
#python merge_json_files.py --json_dir ../output/clevr_4_1/val/ --filename_prefix $FILENAME_PREFIX


##----------------------------------------------------------#

# generate sudoku images
for CLASS_ID in 0
do
time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/clevr_4_1/sudoku/images/ --output_scene_dir ../output/clevr_4_1/sudoku/scenes/ --output_scene_file ../output/clevr_4_1/sudoku/Clevr_4_scenes_sudoku.json --filename_prefix $FILENAME_PREFIX --max_retries $MAX_RETRIES --num_images $NUM_SUDOKU_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 128 --height 128 --properties_json data/properties_Clevr_4.json --conf_class_combos_json data/Clevr_4_GTClasses.json --gt_class_combos_json data/Clevr_4_GTClasses.json --img_class_id $CLASS_ID
done

# merge all classes join files to one json file
python merge_json_files.py --json_dir ../output/clevr_4_1/sudoku/ --filename_prefix $FILENAME_PREFIX
