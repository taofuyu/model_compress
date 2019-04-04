#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --weight_file=./models/yolov3-spp.h5 \
    --deploy_file=./models/yolov3-spp.cfg \
    --save_path=./pruninged_models/ \
    --sensitivity=0.25 \
    --test_imgs_path=./test_imgs/
