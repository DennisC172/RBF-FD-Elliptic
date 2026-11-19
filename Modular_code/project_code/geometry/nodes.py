import numpy as np

class NodeSet:
    points: np.ndarray

    def size(self):
        return self.points.shape[0]

    def dimension(self):
        return self.points.shape[1]