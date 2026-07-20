import numpy as np
import assembly_vec as assembly
import error_analysis
import stencils
from domain import PDEDomainContext
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def min_node_spacing(P, verbose=False):
    tree = KDTree(P)
    dists, idx = tree.query(P, k=2)   # first neighbor is the point itself

    if verbose:
        i = np.argmin(dists[:, 1])
        j = idx[i, 1]
        print("Closest nodes:", i, j, "distance:", dists[i, 1])
        print(P[i], P[j])
    
    return np.min(dists[:, 1])

def operator_gradient_monitor(grad_u, alpha, A=None, d=2, q=2, reg=1e-10):
    """
    Monitor combining physical operator A with solution gradient,
    measured in the A^{-1}-energy norm. If A  is None, then this
    is the regular gradient_monitor function
    """
    
    if A is not None:
        M = A + reg * np.eye(d)[None, :, :]
        M = np.linalg.inv(M)
    else:
        M = np.eye(d)[None, :, :]

    Ainv_gu = np.einsum('nij,nj->ni', M, grad_u)          # A^{-1} grad_u
    energy_norm2 = np.einsum('ni,ni->n', grad_u, Ainv_gu) # grad_u^T A^{-1} grad_u

    prefac = (1 + alpha * energy_norm2) ** (-1.0 / (d + q))
    outer = np.einsum('ni,nj->nij', Ainv_gu, Ainv_gu)

    return prefac[:, None, None] * (M + alpha * outer)

def smooth_monitor(M, S, beta=0.25):
    """
    Smooth a tensor-valued monitor over the stencil graph.

    Parameters
    ----------
    M : (N,d,d)
        Monitor tensor.
    S : (N,k)
        Neighbor indices.
    beta : float
        Smoothing strength.
    """
    avg = M[S].mean(axis=1)
    return (1 - beta) * M + beta * avg

def redistribute_nodes(P, u, num_stencil_nodes, num_rings, basis,
                        shape, L, btype_all_dirichlet, augmentation,
                        A, alpha, eps, tol, sparse, relax=1.0):
    dim = P.shape[1]

    # 1) gradient of current solution on current nodes
    S = stencils.knn_list(P, num_stencil_nodes)
    ctx_g = PDEDomainContext(P, S, A)
    assembly.set_rbf_func(num_rings, basis, augmentation, eps, tol, context=ctx_g)
    grad_u = np.column_stack([Wl @ u for Wl in assembly.global_grads_sparse(ctx_g)])

    # 2) monitor function from that gradient
    M = operator_gradient_monitor(grad_u, alpha, d=dim)
    M = smooth_monitor(M, S, beta=0.5)

    # 3) solve coordinate PDEs, A = M, f = 0, Dirichlet = identity on boundary
    f_zero = lambda p, A: 0.0
    g_x = [
        lambda x: x,   # y=0: x varies
        lambda x: L,   # x=L: x fixed
        lambda x: x,   # y=L: x varies
        lambda x: 0.0  # x=0: x fixed
    ]

    g_y = [
        lambda y: 0.0, # y=0
        lambda y: y,   # x=L
        lambda y: L,   # y=L
        lambda y: y    # x=0
    ]

    ctx_x = assembly.rbf_fd_system(f_zero, g_x, btype_all_dirichlet, P, basis, shape, L,
                                   num_stencil_nodes, num_rings, augmentation,
                                   M, eps, tol, sparse)
    x_new = assembly.rbf_fd_solve_sparse(ctx_x.W, ctx_x.F)

    # reuse the SAME W, just a new rhs for the y-coordinate solve
    g, in_boundary, normal_vec = assembly.set_boundary_func(g_y, btype_all_dirichlet, shape, L, ctx_x)
    f_vec_y = assembly.right_hand_side(ctx_x, f_zero, g, in_boundary)
    y_new = assembly.rbf_fd_solve_sparse(ctx_x.W, f_vec_y)

    P_target = np.column_stack([x_new, y_new])
    spacing_old = min_node_spacing(P)

    P_solved = P.copy()

    while relax > 1e-4:
        P_trial = relax * P_target + (1 - relax) * P

        if min_node_spacing(P_trial) > 0.5 * spacing_old:
            P_solved = P_trial
            break

        relax *= 0.5

    return P_solved, spacing_old # under-relax to avoid overshoot/tangling

def mesh_refinement(f, g, btype, P, rbf_shape, shape, L, num_stencil_nodes,
                    num_rings, augmentation, eig_1, eig_2, angle, eps, tol,
                    sparse=True, max_iters=20):
    
    btype_all_dirichlet = ['dirichlet'] * 4
    
    for it in range(max_iters):
        print(f"Refinement iteration: {it}")
        print("Solving u:")
        A = assembly.coeff_matrix(P.T, eig_1, eig_2, angle)
        ctx = assembly.rbf_fd_system(f, g, btype, P, rbf_shape, shape, L,
                            num_stencil_nodes, num_rings, augmentation,
                            A, eps, tol, sparse)
        u = assembly.rbf_fd_solve_sparse(ctx.W, ctx.F)

        print("Solving P:")
        P_new, spacing_old = redistribute_nodes(P, u, num_stencil_nodes,
                                   num_rings, rbf_shape, shape, L,
                                   btype_all_dirichlet, augmentation,
                                   A, 1e-4, eps, tol, sparse, relax=0.10)

        spacing_new = min_node_spacing(P_new, verbose=True)

        if spacing_new < 0.5 * spacing_old:
            print("Mesh update rejected: node spacing deteriorated.")
            break

        print("Max move:", np.max(np.linalg.norm(P_new-P,axis=1)))

        if np.max(np.linalg.norm(P_new - P, axis=1)) < 1e-7:
            P = P_new
            break
        
        P = P_new   # stencils get rebuilt next iteration automatically
        print()

    plt.scatter(P[:,0], P[:,1], s=1)
    plt.axis("equal")
    plt.show()

    return P
