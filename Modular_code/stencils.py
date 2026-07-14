 # -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:45:33 2026

@author: zeonv

Defines the stencils for the Finite Difference Approximation Scheme.
"""

import numpy as np
from scipy.spatial import cKDTree

def knn_list(P, k):
    tree = cKDTree(P)
    _, idx = tree.query(P, k=k)
    return list(idx)

def knn(P, i, k):
    tree = cKDTree(P)
    _, idx = tree.query(P[i], k=k)
    return idx

# Check proximity of nodes radially
def radial_proximity(P,r):
    C = []

    for k in range(len(P)):
        C.append([])
        for i in range(len(P)):
            if np.linalg.norm(P[k]-P[i]) <= r:
                C[k].append(i)
                
    return C
