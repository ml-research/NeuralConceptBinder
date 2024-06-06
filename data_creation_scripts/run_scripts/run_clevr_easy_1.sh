#!/bin/bash

blender=/usr/bin/blender-2.78c-linux-glibc219-x86_64/blender

#----------------------------------------------------------#
NUM_TRAIN_SAMPLES=10000
NUM_VAL_SAMPLES=2500
NUM_TEST_SAMPLES=2500

NUM_PARALLEL_THREADS=20
NUM_THREADS=4
MIN_OBJECTS=1
MAX_OBJECTS=1
MAX_RETRIES=30

FILENAME_PREFIX=CLEVR_Easy_1
#----------------------------------------------------------#

# generate training images
for CLASS_ID in 0
do
time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/clevr_easy_1/train/images/ --output_scene_dir ../output/clevr_easy_1/train/scenes/ --output_scene_file ../output/clevr_easy_1/train/CLEVR_Easy_1_scenes_train.json --filename_prefix $FILENAME_PREFIX --max_retries $MAX_RETRIES --num_images $NUM_TRAIN_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 128 --height 128 --properties_json data/properties_Clevr_Easy_1.json --conf_class_combos_json data/Clevr_Easy_1_GTClasses.json --gt_class_combos_json data/Clevr_Easy_1_GTClasses.json --img_class_id $CLASS_ID
done

# merge all classes join files to one json file
python merge_json_files.py --json_dir ../output/clevr_easy_1/train/ --filename_prefix $FILENAME_PREFIX

#----------------------------------------------------------#

# generate test images
for CLASS_ID in 0
do
time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/clevr_easy_1/test/images/ --output_scene_dir ../output/clevr_easy_1/test/scenes/ --output_scene_file ../output/clevr_easy_1/test/CLEVR_Easy_1_scenes_test.json --filename_prefix $FILENAME_PREFIX --max_retries $MAX_RETRIES --num_images $NUM_TEST_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 128 --height 128 --properties_json data/properties_Clevr_Easy_1.json --conf_class_combos_json data/Clevr_Easy_1_GTClasses.json --gt_class_combos_json data/Clevr_Easy_1_GTClasses.json --img_class_id $CLASS_ID
done


# merge all classes join files to one json file
python merge_json_files.py --json_dir ../output/clevr_easy_1/test/ --filename_prefix $FILENAME_PREFIX

#----------------------------------------------------------#

# generate confounded val images
for CLASS_ID in 0 1 2
do
time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/clevr_easy_1/val/images/ --output_scene_dir ../output/clevr_easy_1/val/scenes/ --output_scene_file ../output/clevr_easy_1/val/CLEVR_Easy_1_scenes_val.json --filename_prefix $FILENAME_PREFIX --max_retries $MAX_RETRIES --num_images $NUM_VAL_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 128 --height 128 --properties_json data/properties_Clevr_Easy_1.json --conf_class_combos_json data/Clevr_Easy_1_GTClasses.json --gt_class_combos_json data/Clevr_Easy_1_GTClasses.json --img_class_id $CLASS_ID
done

# merge all classes join files to one json file
python merge_json_files.py --json_dir ../output/clevr_easy_1/val/ --filename_prefix $FILENAME_PREFIX
