from sklearn.cluster import KMeans
import numpy as np
from numba import jit,prange

@jit(parallel=True)
def kmean_img(im):
    x,y = im.shape
    kmean_input = []

    for i in prange(x):
        for j in prange(y):
            kmean_input.append(np.array([i,j,im[i,j]]))
    
    kmeans = KMeans(n_clusters=3)
    bined = kmeans.fit_predict(kmean_input)

    new_img = np.zeros(cleaned.shape)

    for j in prange(len(kmean_input)):
        index =  kmean_input[j][0:2]
        val = bined[j]
        new_img[int(index[0]),int(index[1])] = val
    
    return im