import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix,csr_matrix
from keras.models import load_model

#input for csr_matrix should be 2-dim,but raw weights is 4-dim(w,h,c,n)
#shape of dim_2_list is (kernel's width, height*channel*num)
def cvt_dim(layer_weight):
    shape = layer_weight.shape
    dim_2_list = np.zeros(shape=(shape[0],shape[1]*shape[2]*shape[3]))
    all_ele = layer_weight.flatten('F')
    count=0
    for i in range(dim_2_list.shape[1]):
        for j in range(dim_2_list.shape[0]):
            dim_2_list[j,i] = all_ele[count]
            count = count + 1
    
    return dim_2_list

#after Kmeans, clustered non-zero weights are saved in an one col array
#therefore, restore this array to (w,h,c,n) format with zero-weights.
def reset_weights(raw_weights, kmeans_weights):
    shape = raw_weights.shape
    count = 0
    for r in range(shape[0]):
        for n in range(shape[3]):
            for c in range(shape[2]):
                for l in range(shape[1]):
                    #replace the non-zero raw weight with corresponding clustered weight 
                    if raw_weights[r,l,c,n]:
                        raw_weights[r,l,c,n] = kmeans_weights[count]
                        count = count + 1
    return raw_weights

def share_weights(yolo_obj, bits):
    for layer in yolo_obj.yolo_model.layers:
        if layer.name.split('_')[0] == 'conv2d':
            layer_weights = layer.get_weights()
            for i in range(len(layer_weights)):
                dim2_layer_weights = cvt_dim(layer_weights[i])
                shape = dim2_layer_weights.shape
                #use csc/csr mat to represent sparse weights
                if shape[0] < shape[1]: #h<w. i.e. rows are small, the JA matrix will be short
                    sparse_mat = csr_matrix(dim2_layer_weights)
                else:
                    sparse_mat = csc_matrix(dim2_layer_weights)
                min_val = min(sparse_mat.data)
                max_val = max(sparse_mat.data)
                #use min/max value to create a linspace. 
                #n bits means this linspace will be set to 2**n segments between [min,max]
                linspace = np.linspace(min_val, max_val, num=2**bits)
                #use linspace to set cluster centers
                Kmeans = KMeans(n_clusters=len(linspace), init=linspace.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
                #perform Kmeans proc
                Kmeans.fit(sparse_mat.data.reshape(-1,1))
                #obtain Kmeans result
                new_weights = Kmeans.cluster_centers_[Kmeans.labels_].reshape(-1)

                layer_weights[i] = reset_weights(layer_weights[i], new_weights)
            layer.set_weights(layer_weights)
    return yolo_obj




        
    