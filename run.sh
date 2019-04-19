#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --save_path=./saves/ \
    --sensitivity=0.25 \
    --raw_model_path=./models/trained_weights_stage_1.h5 \
    --test_img_path=./test_imgs/
