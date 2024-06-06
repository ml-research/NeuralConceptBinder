#!/bin/bash

blender=/usr/bin/blender-2.78c-linux-glibc219-x86_64/blender

#----------------------------------------------------------#
NUM_TRAIN_SAMPLES=2000
NUM_VAL_SAMPLES=500
NUM_TEST_SAMPLES=500

NUM_PARALLEL_THREADS=5
NUM_THREADS=4
MIN_OBJECTS=3
MAX_OBJECTS=4
MAX_RETRIES=30

#----------------------------------------------------------#

# generate training images
for CLASS_ID in 0 1 2
do
time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/clevrhans3_4objs/train/images/ --output_scene_dir ../output/clevrhans3_4objs/train/scenes/ --output_scene_file ../output/clevrhans3_4objs/train/CLEVR_HANS_4objs_scenes_train.json --filename_prefix CLEVR_Hans_4objs --max_retries $MAX_RETRIES --num_images $NUM_TRAIN_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 128 --height 128 --properties_json data/properties_Clevr_4.json --conf_class_combos_json data/Clevr_Hans_ConfClasses_3_4objs.json --gt_class_combos_json data/Clevr_Hans_GTClasses_3_4objs.json --img_class_id $CLASS_ID
done

# merge all classes join files to one json file
python merge_json_files.py --json_dir ../output/clevrhans3_4objs/train/ --filename_prefix CLEVR_Hans_4objs

#----------------------------------------------------------#

# generate test images
for CLASS_ID in 0 1 2
do
time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/clevrhans3_4objs/test/images/ --output_scene_dir ../output/clevrhans3_4objs/test/scenes/ --output_scene_file ../output/clevrhans3_4objs/test/CLEVR_HANS_4objs_scenes_test.json --filename_prefix CLEVR_Hans_4objs --num_images $NUM_TEST_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 128 --height 128 --properties_json data/properties_Clevr_4.json --conf_class_combos_json data/Clevr_Hans_GTClasses_3_4objs.json --gt_class_combos_json data/Clevr_Hans_GTClasses_3_4objs.json --img_class_id $CLASS_ID
done

# merge all classes join files to one json file
python merge_json_files.py --json_dir ../output/clevrhans3_4objs/test/ --filename_prefix CLEVR_Hans_4objs

#----------------------------------------------------------#

# generate confounded val images
for CLASS_ID in 0 1 2
do
time $blender --threads $NUM_THREADS --background -noaudio --python render_images_clevr_hans.py -- --output_image_dir ../output/clevrhans3_4objs/val/images/ --output_scene_dir ../output/clevrhans3_4objs/val/scenes/ --output_scene_file ../output/clevrhans3_4objs/val/CLEVR_HANS_4objs_scenes_val.json --filename_prefix CLEVR_Hans_4objs --num_images $NUM_VAL_SAMPLES --min_objects $MIN_OBJECTS --max_objects $MAX_OBJECTS --num_parallel_threads $NUM_PARALLEL_THREADS --width 128 --height 128 --properties_json data/properties_Clevr_4.json --conf_class_combos_json data/Clevr_Hans_ConfClasses_3_4objs.json --gt_class_combos_json data/Clevr_Hans_GTClasses_3_4objs.json --img_class_id $CLASS_ID
done

# merge all classes join files to one json file
python merge_json_files.py --json_dir ../output/clevrhans3_4objs/val/ --filename_prefix CLEVR_Hans_4objs