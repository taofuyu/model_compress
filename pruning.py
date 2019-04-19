import os
import numpy as np
from keras.models import load_model
import util
from yolo import YOLO

def pruning(yolo_obj, sensitivity):
    for layer in yolo_obj.yolo_model.layers:
        if layer.name.split('_')[0] == 'conv2d':
            print('conv_layer: '+layer.name.split('_')[1])
            layer_weights = layer.get_weights()
            #print(layer_weights)
            #print(type(layer_weights))
            for i in range(len(layer_weights)):
                threshold = sensitivity * np.std(layer_weights[i])
                layer_weights[i] = np.where(np.abs(layer_weights[i])<threshold,0,layer_weights[i])
            #print(layer_weights)
            layer.set_weights(layer_weights)

    return yolo_obj

