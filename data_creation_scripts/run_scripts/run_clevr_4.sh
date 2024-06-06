#!/bin/bash

blender=/usr/bin/blender-2.78c-linux-glibc219-x86_64/blender

#----------------------------------------------------------#
NUM_TRAIN_SAMPLES=175000
#NUM_VAL_SAMPLES=5000
NUM_TEST_SAMPLES=11000

NUM_PARALLEL_THREADS=20
NUM_THREADS=4
MIN_OBJECTS=1
MAX_OBJECTS=4
MAX_RETRIES=30

FILENAME_PREFIX=CLEVR_4
#----------------------------------------------------------#

# generate training images
for CLASS_ID in 0
do
time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/clevr_4/train/images/ --output_scene_dir ../output/clevr_4/train/scenes/ --output_scene_file ../output/clevr_4/train/Clevr_4_scenes_train.json --filename_prefix $FILENAME_PREFIX --max_retries $MAX_RETRIES --num_images $NUM_TRAIN_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 128 --height 128 --properties_json data/properties_Clevr_4.json --conf_class_combos_json data/Clevr_4_GTClasses.json --gt_class_combos_json data/Clevr_4_GTClasses.json --img_class_id $CLASS_ID
done

# merge all classes join files to one json file
python merge_json_files.py --json_dir ../output/clevr_4/train/ --filename_prefix $FILENAME_PREFIX

##----------------------------------------------------------#
#
## generate test images
#for CLASS_ID in 0
#do
#time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/clevr_4/test/images/ --output_scene_dir ../output/clevr_4/test/scenes/ --output_scene_file ../output/clevr_4/test/Clevr_4_scenes_test.json --filename_prefix $FILENAME_PREFIX --max_retries $MAX_RETRIES --num_images $NUM_TEST_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 128 --height 128 --properties_json data/properties_Clevr_4.json --conf_class_combos_json data/Clevr_4_GTClasses.json --gt_class_combos_json data/Clevr_4_GTClasses.json --img_class_id $CLASS_ID
#done
#
#
## merge all classes join files to one json file
#python merge_json_files.py --json_dir ../output/clevr_4/test/ --filename_prefix $FILENAME_PREFIX
#
##----------------------------------------------------------#
#
## generate confounded val images
#for CLASS_ID in 0 1 2
#do
#time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/clevr_4/val/images/ --output_scene_dir ../output/clevr_4/val/scenes/ --output_scene_file ../output/clevr_4/val/Clevr_4_scenes_val.json --filename_prefix $FILENAME_PREFIX --max_retries $MAX_RETRIES --num_images $NUM_VAL_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 128 --height 128 --properties_json data/properties_Clevr_4.json --conf_class_combos_json data/Clevr_4_GTClasses.json --gt_class_combos_json data/Clevr_4_GTClasses.json --img_class_id $CLASS_ID
#done
#
## merge all classes join files to one json file
#python merge_json_files.py --json_dir ../output/clevr_4/val/ --filename_prefix $FILENAME_PREFIX
