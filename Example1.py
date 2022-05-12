# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:59:33 2022

@author: gulrch
"""

from rs import rs
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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
    
    #plotting the kmeans results
    for j in np.unique(km.labels_):
         plt.scatter(X[km.labels_ == j , 0] , X[km.labels_ == j , 1] , label = j)
    plt.scatter(km.cluster_centers_[:,0] , km.cluster_centers_[:,1] , s = 80, color = 'k')
    # displaying the title
    plt.title("kmeans results of iteration: "+str(i))
    plt.show()
    
    #plotting the random swap results
    for j in np.unique(randomSwap.labels_):
         plt.scatter(X[randomSwap.labels_ == j , 0] , X[randomSwap.labels_ == j , 1] , label = j)
    plt.scatter(randomSwap.cluster_centers_[:,0] , randomSwap.cluster_centers_[:,1] , s = 80, color = 'k')
    # displaying the title
    plt.title("random swap results of iteration: "+str(i))
    plt.show()
    
    
    
    
