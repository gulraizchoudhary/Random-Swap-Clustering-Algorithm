# 
# Random Swap 
# G. I. Choudhary
# 24.4.2022
# common parameters:
# X: data set
# C: centroids

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import random as rand


__version__="V1.1"

class rs(KMeans):
    def get_version():
        return __version__
    
    def __init__(self, n_init=1, **kwargs):
        """ n_init: number of times k-means run initially
            kwargs: arguments for scikit-learns KMeans """
            
        super().__init__(n_init=n_init,  init='random', **kwargs)

    def _kmeans(self,C,X):
        """perform kmeans algorithm"""
        self.init = C # set cluster centers
        self.n_clusters = len(C) # set k-value
        super().fit(X) # kmeans algorithm

    def fit(self, X, iterations):
        """ compute random swap using random removal and random addition"""
        
        # run k-means
        super().fit(X) # requires self.n_clusters >= 1
        
        # memorize best error and codebook so far
        E_best = self.inertia_
        C_best = self.cluster_centers_
        l_best = self.labels_
        
        tmp = self.n_init, self.init # store for compatibility with sklearn
        
        for i in range(0,iterations):
            C = self.cluster_centers_
            C[rand.choice(range(0,len(C)))] = X[rand.randint(0, len(X)-1)]
            self.cluster_centers_ = C
            
            self._kmeans(C,X)
            
            if self.inertia_ < E_best*(1-self.tol):
                # improvement! update memorized best error and codebook so far
                E_best = self.inertia_
                C_best = self.cluster_centers_
                l_best = self.labels_
                
                
                
        self.n_init, self.init = tmp # restore for compatibility with sklearn
        self.inertia_ = E_best
        self.cluster_centers_ = C_best
        self.labels_ = l_best
            
                
        return self