# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:59:33 2022

@author: gulrch
"""

from rs import rs
import numpy as np
from sklearn.cluster import KMeans

#random 2D data set
X=np.random.rand(1000,2)

# number of centroids
k=100
swaps = 1000

for i in range(5):
    randomSwap = rs(n_clusters=k)
    randomSwap.fit(X, swaps)

    km = KMeans(n_clusters=k, init='random').fit(X)
    
    # relative SSE improvement of random swap over kmeans
    imp = 1 - randomSwap.inertia_/km.inertia_
    print(f"SSE improvement over k-means: {imp:.2%}")
    
    
