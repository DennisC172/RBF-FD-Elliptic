import numpy as np
from scipy.spatial import cKDTree

class StencilSet:
    indices: np.ndarray

    def size(self):
        return self.indices.shape[0]