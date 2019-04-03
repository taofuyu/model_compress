import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix,csr_matrix

def share_weights(model,bits):
    for layer in model.layers:
        
    