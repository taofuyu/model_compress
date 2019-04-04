import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix,csr_matrix
from keras.models import load_model
import pruning

def share_weights(model,bits):
    for layer in model.layers:
        if layer.name.split('_')[0] == 'conv2d':
            layer_weights = layer.get_weights()
            for i in range(len(layer_weights)):
                shape = layer_weights[i].shape
                #use csc/csr mat to represent sparse weights
                if shape[0] < shape[1]: #h<w. i.e. rows are small, the JA matrix will be short
                    sparse_mat = csr_matrix(layer_weights[i])
                else:
                    sparse_mat = csc_matrix(layer_weights[i])
                min_val = min(sparse_mat.data)
                max_val = max(sparse_mat.data)
                #use min/max value to create a linspace. 
                #n bits means this linspace will be set to 2**n segments between [min,max]
                linspace = np.linspace(min_val, max_val, num=2**bits)
                print(linspace)
                #use linspace to create a KMeans object
                Kmeans = KMeans(n_clusters=len(linspace), init=linspace.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
                #perform Kmeans proc
                Kmeans.fit(sparse_mat.data.reshape(-1,1))
                #obtain Kmeans result
                new_weights = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)

                layer_weights[i] = new_weights
            layer.set_weights(layer_weights)
    return model

if __name__ == '__main__':
    model = load_model('/home/tao/Downloads/pruning/models/yolov3-spp.h5')
    model = pruning.pruning(model,0.25)##save this model to truly change weights
    model = share_weights(model,5)




        
    