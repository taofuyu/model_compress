# model_compress
model compress, Keras, YOLOv3
Keras implementation of 'Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding'. Specifically, it's implemented for keras version of YOLOv3.

## Introduction

1. Train own keras model. I use [keras-yolo3-master](https://github.com/qqwweee/keras-yolo3)
    Then get the .h5 weight file.
2. Perform compression proc, i.e., this project.
3. Use [mAP-master](https://github.com/Cartucho/mAP) to calculate mAP.

THIS PROJECT IS NOT FINISHED YET!

## Quick Start

1. After obtaining trained model, put it in folder ./models/, and modify the param 'raw_model_path' in run.sh

2. Copy test images in ./test_imgs/

3. Put class file and anchor file in ./models/. Then modify corresponding params _defaults in yolo.py 

4. Run this project: bash run.sh

## Details

1. run.sh will call main.py. In main.py, it first uses trained model to detect test imgs, then saves detection results and drawed imgs in ./saves/raw_model/. Dtection results are saved as a .txt file, whose name is the same as corresponding img name(i.e., detection result for dog.png will be saved in dog.txt). In this .txt file, format of each line: class_name, detect_score, left, top, bottom, right. In furthuer proc, this .txt file can be used to cal mAP.

2. In main.py, it second prunings trained model to obtain pruning_model, then save the weights of pruning_model and use pruning_model to detect test imgs as in Details 1.

3. After obtaining pruning_model, the quantization proc is applied. Then use quan_model to detect test imgs as in Details 1.

## Cal mAP

To cal mAP, copy the detection results, i.e. all the .txt files, to folder 'input/detection-results/' in [mAP-master](https://github.com/Cartucho/mAP). Then follow the README in [mAP-master](https://github.com/Cartucho/mAP)
