import argparse
import os
import util
from yolo import YOLO
from pruning import pruning
from quantizes_weight import share_weights

###############################################################################################################
#
#               size/mAP                  size/mAP                   size/mAP               size/mAP
# trained model----------> pruning model----------> quantized model----------> final model-----------|
#                pruning                  quantize                    huffman
#
################################################################################################################

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path',default='./saves/')
    parser.add_argument('--sensitivity',default=0.25)
    parser.add_argument('--test_img_path',default='./test_imgs/')
    parser.add_argument('--raw_model_path')


    args = parser.parse_args()

    #init yolo object. Its members include default model file\anchor file\class file
    #to use own params above, pass **kwargs to YOLO()
    raw_yolo = YOLO(model_path=args.raw_model_path)
    util.detect_and_save(raw_yolo, args.test_img_path, args.save_path+'raw_model/')
    #pruning and save
    if True:
        pruning_yolo = pruning(raw_yolo, args.sensitivity)
        #use YOLO_OBJ.yolo_model to get corresponding weights
        pruning_yolo.yolo_model.save('models/model_after_pruning.h5')
        #to retrain and reload model.......
        util.detect_and_save(pruning_yolo, args.test_img_path, args.save_path+'prun_model/')
        
    #quantizes weight by sharing weights
    if True:
        quantized_yolo = share_weights(pruning_yolo)
        quantized_yolo.yolo_model.save('models/model_after_quan.h5')
        util.detect_and_save(quantized_yolo, args.test_img_path, args.save_path+'quan_model/')
        
if __name__ == '__main__':
    _main()