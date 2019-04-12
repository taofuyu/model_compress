import pruning
import quantizes_weight
#import huffman_code
import argparse
import os
import util
from keras.models import load_model

###############################################################################################################
#
#               size/mAP                  size/mAP                   size/mAP               size/mAP
# trained model----------> pruning model----------> quantized model----------> final model-----------|
#                pruning                  quantize                    huffman
#
################################################################################################################

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file')
    parser.add_argument('--deploy_file')
    parser.add_argument('--save_path')
    parser.add_argument('--sensitivity')
    parser.add_argument('--test_imgs_path')
    parser.add_argument('--gt_bb_class_file')
    parser.add_argument('--log_file',default='./log/log.txt')

    args = parser.parse_args()

    #num_classes = len()
    model = load_model(args.weight_file)
    model_out = model.metrics_name
    mAP = util.compute_mAP()
    util.log(args.log_file, 'raw mAP: {}'.format(mAP))
    util.log(args.log_file, 'raw model size: {}'.format(str(os.path.getsize(model))))
    #pruning and save
    if True:
        pruning_model = pruning.pruning(model, args.sensitivity)
        #to retrain .......
        mAP = util.compute_mAP()
        util.log(args.log_file, 'mAP after pruning: {}'.format(mAP))
        pruning_model.save(save_path+'yolov3-spp-pruning.h5')
        util.log(args.log_file, 'pruning model size: {}'.format(str(os.path.getsize(save_path+'yolov3-spp-pruning.h5'))))
    
    #quantizes weight by sharing weights
    if True:
        quantized_model = quantizes_weight.share_weights(pruning_model)
        

if __name__ == '__main__':
    _main()