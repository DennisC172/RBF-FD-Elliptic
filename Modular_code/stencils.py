 # -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:45:33 2026

@author: zeonv

Defines the stencils for the Finite Difference Approximation Scheme.
"""

import numpy as np

# KMeans generated for index i
def knn(P, i, k):
    dist = np.linalg.norm(P - P[i], axis=1)
    idx = np.argsort(dist)
    return idx[:k]

# KMeans generated for all ndoes
def knn_list(P, k):
    S = []
    
    for i in range(len(P)):
        S.append(knn(P,i,k))
        
    return S

# Check proximity of nodes radially
def radial_proximity(P,r):
    C = []

    for k in range(len(P)):
        C.append([])
        for i in range(len(P)):
            if np.linalg.norm(P[k]-P[i]) <= r:
                C[k].append(i)
                
    return C
