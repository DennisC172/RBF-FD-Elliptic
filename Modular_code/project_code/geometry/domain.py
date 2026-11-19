import stencils.py

def knn_list(P, k):
    tree = stencils.cKDTree(P)
    _, idx = tree.query(P, k=k)
    return list(idx)

def knn(P, i, k):
    tree = stencils.cKDTree(P)
    _, idx = tree.query(P[i], k=k)
    return idx
    

class Domain:
    def __init__(self, geometry, boundary):
        self.geometry = geometry
        self.boundary = boundary

    def generate_nodes(self, resolution):
        pass

    def generate_stencils(self, nodes, stencil_size):
        return knn_list(nodes, stencil_size)

    