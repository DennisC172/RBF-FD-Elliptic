# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:04:22 2026

@author: Dennis Corraliza

Provides the assembly methods to solve FD Methods
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from domain import PDEDomainContext
import stencils
import rbf_vec as rbf
import boundary
import geometry

def plot_stencil_centers(nodes, centers, p_i):
    nodes = np.asarray(nodes)
    centers = np.asarray(centers)
    p_i = np.asarray(p_i)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Stencil nodes
    ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        s=60,
        marker="o",
        label=f"Stencil nodes ({len(nodes)})"
    )

    # RBF centers
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=90,
        marker="x",
        label=f"Centers ({len(centers)})"
    )

    # Evaluation node
    ax.scatter(
        p_i[0],
        p_i[1],
        s=200,
        marker="*",
        label=r"$P_i$"
    )

    # Annotate stencil nodes
    for j, p in enumerate(nodes):
        ax.annotate(
            f"$P_{j}$",
            (p[0], p[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )

    # Annotate centers
    for j, c in enumerate(centers):
        ax.annotate(
            f"$C_{j}$",
            (c[0], c[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("RBF-FD stencil and centers")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()

def local_weights_solve(context, i, eps0, fixed_eps=False):
    s = context.stencils[i]
    P = context.nodes
    num_nodes = len(P)

    # Augmentation form
    pdim = 0
    if context.augmentation:
        pdim = len(rbf.poly_basis(P[0]))

    # M: pairwise phi(P[s[j]] - P[s[k]]) for all (j,k), one vectorized call
    Ps = P[s]
    Ai = context.A[i]
    diff_M = Ps[:, None, :] - Ps[None, :, :]   # (num_nodes, num_nodes, dim)

    if fixed_eps:
        eps = eps0
    else:
        eps = rbf.stencil_eps(diff_M, eps0)

    M = context.phi(diff_M, eps)               # (num_nodes, num_nodes)

    # b: laplacian_phi(P[i] - P[s[k]]) for all k, one vectorized call
    diff_b = P[i][None, :] - Ps                # (num_nodes, dim)
    b = context.laplacian_phi(diff_b, Ai, eps) # (num_nodes,)

    if context.augmentation:
        Pmat = rbf.poly_basis(Ps)                       # (num_nodes, pdim)
        b_poly = rbf.anisotropic_diffusion_poly(P[i])   # (pdim)
        pdim =  Pmat.shape[1]

        Q, R = np.linalg.qr(Pmat, mode="complete")
        Q1 = Q[:,:pdim]
        Q2 = Q[:,pdim:]

        w_1 = Q1 @ np.linalg.solve(R[:pdim].T, b_poly)
        b = Q2.T @ (b - M @ w_1)
        M = Q2.T @ M @ Q2
    else:
        w_1 = np.zeros(num_nodes)

    '''print(f"Conditioning: {np.linalg.cond(M): e}, ({M.shape}) at idx = {i}")
    s = np.linalg.svd(M, compute_uv=False)
    print(s)
    print(s[-1]).
    print(np.linalg.matrix_rank(M))'''
    Q,R = np.linalg.qr(M)
    y = Q.T @ b
    z = np.linalg.solve(R, y)
    return z if not context.augmentation else Q2 @ z + w_1, np.linalg.cond(M)

### Least Squares
def local_weights_ls(context, i, eps0, fixed_eps=False, lam=1.0e-3):
    Ai = context.A[i]
    P = context.nodes

    s = context.stencils[i]
    Ps = P[s]
    num_nodes = len(Ps)

    k = context.centers_info
    dP_norm = np.linalg.norm(Ps - P[i], axis=1)
    r_max = np.max(dP_norm)
    points = geometry.quasi_circle(1. * r_max, k)
    Cs = P[i] + points
    num_centers = len(Cs)

    diff_M = Ps[None, :, :] - Cs[:, None, :]

    if fixed_eps:
        eps = eps0
    else:
        eps = rbf.stencil_eps(diff_M, eps0)

    M = context.phi(diff_M, eps)

    diff_b = P[i][None, :] - Cs
    b = context.laplacian_phi(diff_b, Ai, eps)

    if context.augmentation:
        Pmat = rbf.poly_basis(Ps)
        b_poly = rbf.anisotropic_diffusion_poly(P[i])
        pdim = Pmat.shape[1]

        Q, R = np.linalg.qr(Pmat, mode="complete")
        Q1 = Q[:, :pdim]
        Q2 = Q[:, pdim:]

        w_1 = Q1 @ np.linalg.solve(R[:pdim].T, b_poly)
        b_2 = M @ w_1
        M = M @ Q2
    else:
        Q2 = np.eye(num_nodes)
        w_1 = np.zeros(num_nodes)
        b_2 = np.zeros(num_centers)

    # --------------------------------------------------
    # RKHS residual regularization
    # --------------------------------------------------
    if lam > 0:

        diff_G = Ps[:, None, :] - Ps[None, :, :]
        G = context.phi(diff_G, eps)

        diff_c = P[i][None, :] - Ps
        c = context.laplacian_phi(diff_c, Ai, eps)

        # G = Rg.T @ Rg
        Rg = np.linalg.cholesky(G).T

        # Rg.T @ q = c - G @ w_1
        q = np.linalg.solve(Rg.T, c - G @ w_1)

        M = np.vstack([
            M,
            np.sqrt(lam) * (Rg @ Q2)
        ])

        b_2 = np.concatenate([
            b - b_2,
            np.sqrt(lam) * q
        ])

    else:
        b_2 = b - b_2

    # --------------------------------------------------
    # QR least-squares solve
    # --------------------------------------------------
    Q, R = np.linalg.qr(M, mode="reduced")
    y = Q.T @ b_2
    z = np.linalg.solve(R, y)

    w = z if not context.augmentation else Q2 @ z + w_1

    return w, np.linalg.cond(M)
    
def local_grad_solve(context, i, eps0, fixed_eps=False, add_anisotropy=True):
    At = context.A[i].T
    s = context.stencils[i] 
    P = context.nodes
    num_nodes = len(P)

    # Augmentation form
    pdim = 0
    if context.augmentation:
        pdim = len(rbf.poly_basis(P[0]))

    Ps = P[s]
    # M: pairwise phi(P[s[j]] - P[s[k]]) for all (j,k), one vectorized call
    diff_M = Ps[:, None, :] - Ps[None, :, :]   # (num_nodes, num_nodes, dim)

    if fixed_eps:
        eps = eps0
    else:
        eps = rbf.stencil_eps(diff_M, eps0)

    M = context.phi(diff_M, eps)               # (num_nodes, num_nodes)

    # b_grad: grad_phi(P[i] - P[s[k]]) for all k, one vectorized call
    diff_b = P[i][None, :] - Ps                # (num_nodes, dim)
    b_grad = context.grad_phi(diff_b, eps)     # (num_nodes, dim)

    if add_anisotropy:
        b_grad = b_grad @ At

    if context.augmentation:
        Pmat = rbf.poly_basis(Ps)               # (num_nodes, pdim)
        b_poly = rbf.grad_poly(P[i])            # (pdim, dim)
        pdim =  Pmat.shape[1]

        Q, R = np.linalg.qr(Pmat, mode="complete")
        Q1 = Q[:,:pdim]
        Q2 = Q[:,pdim:]

        w_1 = Q1 @ np.linalg.solve(R[:pdim].T, b_poly)
        b_grad = Q2.T @ (b_grad - M @ w_1)
        M = Q2.T @ M @ Q2

    else:
        dim = P.shape[1]
        w_1 = np.zeros((num_nodes, dim))

    Q,R = np.linalg.qr(M)
    y = Q.T @ b_grad
    z_grad = np.linalg.solve(R, y)
    return z_grad if not context.augmentation else Q2 @ z_grad + w_1, np.linalg.cond(M)

def local_grad_ls(context, i, eps0, fixed_eps=False, add_anisotropy=True, lam=1.0e-3):
    At = context.A[i].T
    P = context.nodes

    s = context.stencils[i]
    Ps = P[s]
    num_nodes = len(Ps)

    k = context.centers_info
    dP_norm = np.linalg.norm(Ps - P[i], axis=1)
    r_max = np.max(dP_norm)
    points = geometry.quasi_circle(1. * r_max, k)
    Cs = P[i] + points
    num_centers = len(Cs)

    diff_M = Ps[None, :, :] - Cs[:, None, :]

    if fixed_eps:
        eps = eps0
    else:
        eps = rbf.stencil_eps(
            Ps[None, :, :] - Ps[:, None, :], eps0
        )

    M = context.phi(diff_M, eps)

    diff_b = P[i][None, :] - Cs
    b_grad = context.grad_phi(diff_b, eps)

    if add_anisotropy:
        b_grad = b_grad @ At

    if context.augmentation:
        Pmat = rbf.poly_basis(Ps)
        b_poly = rbf.grad_poly(P[i])

        if add_anisotropy:
            b_poly = b_poly @ At

        pdim = Pmat.shape[1]

        Q, R = np.linalg.qr(Pmat, mode="complete")
        Q1 = Q[:, :pdim]
        Q2 = Q[:, pdim:]

        w_1 = Q1 @ np.linalg.solve(R[:pdim].T, b_poly)
        b_2 = M @ w_1
        M = M @ Q2

    else:
        dim = P.shape[1]
        Q2 = np.eye(num_nodes)
        w_1 = np.zeros((num_nodes, dim))
        b_2 = np.zeros((num_centers, dim))

    # Add RKHS residual regularization
    if lam > 0:
        diff_G = Ps[:, None, :] - Ps[None, :, :]
        G = context.phi(diff_G, eps)

        diff_c = P[i][None, :] - Ps
        c = context.grad_phi(diff_c, eps)

        if add_anisotropy:
            c = c @ At

        Rg = np.linalg.cholesky(G).T

        q = np.linalg.solve(Rg.T, c - G @ w_1)

        M = np.vstack([
            M,
            np.sqrt(lam) * (Rg @ Q2)
        ])

        b_2 = np.vstack([
            b_grad - b_2,
            np.sqrt(lam) * q
        ])

    else:
        b_2 = b_grad - b_2

    # Solve the augmented least-squares problem using QR
    Q, R = np.linalg.qr(M, mode="reduced")
    y = Q.T @ b_2
    z_grad = np.linalg.solve(R, y)

    w = z_grad if not context.augmentation else Q2 @ z_grad + w_1

    return w, np.linalg.cond(M)

# .2 Assembles the global Weights.
def global_weights(context, eps, fixed_eps=False, in_boundary=None, normal_vec=None):
    P = context.nodes
    num_nodes = len(P)    
    
    S = context.stencils
    k = context.centers_info
    
    W = np.zeros((num_nodes, num_nodes))

    cond_A = 0.0
    idx_c = 0
    node_cond = ""
    
    for i,s in enumerate(S):
        num_stencil_nodes = len(s)
        node_type = in_boundary(P[i]) if in_boundary is not None else 'interior'
        
        if node_type != 'interior':
            if node_type == 'dirichlet':
                W[i,:] = 0.0
                W[i,i] = 1.0
            elif node_type == 'neumann':
                if k is None:
                    w_grad, kA = local_grad_solve(context, i, eps, fixed_eps)
                else:
                    w_grad, kA = local_grad_ls(context, i, eps, fixed_eps) 
                
                n_v = normal_vec(P[i])                
                dir_derv = w_grad @ n_v
                
                for j in range(num_stencil_nodes):
                    W[i,s[j]] = dir_derv[j]
        else:
            if k is None:
                w, kA = local_weights_solve(context, i, eps, fixed_eps)
            else:
                w, kA = local_weights_ls(context, i, eps, fixed_eps)
    
            for j in range(num_stencil_nodes):
                W[i,s[j]] = w[j]

        if kA > cond_A:
            idx_c = i
            cond_A = kA
            node_cond = node_type

    print(f"Max conditioning: {cond_A: e} at {idx_c} ({node_cond})")     
    return W

def global_weights_sparse(context, eps, fixed_eps=False, in_boundary=None, normal_vec=None):
    P = context.nodes
    num_nodes = len(P)
    
    S = context.stencils
    k = context.centers_info
    
    max_nnz = num_nodes * context.stencils[0].shape[0]  # upper bound

    rows = np.empty(max_nnz, dtype=np.int32)
    cols = np.empty(max_nnz, dtype=np.int32)
    vals = np.empty(max_nnz, dtype=np.float64)
    ptr = 0

    cond_A = 0.0
    idx_c = 0
    node_cond = ""
    
    for i,s in enumerate(S):
        node_type = in_boundary(P[i]) if in_boundary is not None else 'interior'
        kA = 0.0

        if node_type == 'dirichlet':
            rows[ptr] = i
            cols[ptr] = i
            vals[ptr] = 1.0
            ptr += 1
        elif node_type == 'neumann':
            if k is None:
                w_grad, kA = local_grad_solve(context, i, eps, fixed_eps)                
            else:
                w_grad, kA = local_grad_ls(context, i, eps, fixed_eps)

            n_v = normal_vec(P[i])
            dir_derv = w_grad @ n_v

            n = len(s)
            rows[ptr:ptr+n] = i
            cols[ptr:ptr+n] = s
            vals[ptr:ptr+n] = dir_derv
            ptr += n
        else:
            if k is None:
                w, kA = local_weights_solve(context, i, eps, fixed_eps)
            else:
                w, kA = local_weights_ls(context, i, eps, fixed_eps)

            n = len(s)
            rows[ptr:ptr+n] = i
            cols[ptr:ptr+n] = s
            vals[ptr:ptr+n] = w
            ptr += n
        
        if kA > cond_A:
            idx_c = i
            cond_A = kA
            node_cond = node_type
            
    W = sp.coo_matrix((vals[:ptr], (rows[:ptr], cols[:ptr])),
                      shape=(num_nodes, num_nodes))

    print(f"Max conditioning: {cond_A: e} at {P[idx_c]} ({node_cond})")      
    return W.tocsr()

def global_grads(context, eps, fixed_eps=False):
    P = context.nodes
    num_nodes = len(P)  
    dim = len(P[0])
    
    S = context.stencils
    k = context.centers_info
    
    W = np.zeros((num_nodes, num_nodes, dim))
    
    for i,s in enumerate(S):
        num_stencil_nodes = len(s)
        
        if k is None:
            w_grad, _ = local_grad_solve(context, i, eps, fixed_eps)
        else:
            w_grad, _ = local_grad_ls(context, i, eps, fixed_eps) 
            
        for j in range(num_stencil_nodes):
            for l in range(dim):
                W[i,s[j],l] = w_grad[j,l]
            
    return W

def global_grads_sparse(context, eps, fixed_eps=False):
    P = context.nodes
    num_nodes = len(P)
    dim = len(P[0])

    S = context.stencils
    k = context.centers_info

    max_stencil = max(len(s) for s in S)
    max_nnz = num_nodes * max_stencil

    rows = np.empty(max_nnz, dtype=np.int64)
    cols = np.empty(max_nnz, dtype=np.int64)
    vals = np.empty((dim, max_nnz), dtype=np.float64)
    ptr = 0

    for i, s in enumerate(S):
        n = len(s)

        if k is None:
            w_grad, _ = local_grad_solve(context, i, eps, fixed_eps)
        else:
            w_grad, _ = local_grad_ls(context, i, eps, fixed_eps)

        rows[ptr:ptr+n] = i
        cols[ptr:ptr+n] = s
        for l in range(dim):
            vals[l, ptr:ptr+n] = w_grad[:, l]
        ptr += n

    W_list = [
        sp.coo_matrix((vals[l, :ptr], (rows[:ptr], cols[:ptr])),
                      shape=(num_nodes, num_nodes)).tocsr()
        for l in range(dim)
    ]

    return W_list

def boundary_to_weights(W, context, in_boundary, normal_vec, eps, fixed_eps=False):
    P = context.nodes   
    
    S = context.stencils
    k = context.centers_info
    
    W_b = W.copy()
    
    for i,s in enumerate(S):
        num_stencil_nodes = len(s)
        node_type = in_boundary(P[i])
        
        if node_type != 'interior':
            if node_type == 'dirichlet':
                W_b[i,:] = 0.0
                W_b[i,i] = 1.0
            elif node_type == 'neumann':
                if k is None:
                    w_grad = local_grad_solve(context, i, eps, fixed_eps)
                else:
                    w_grad = local_grad_ls(context, i, eps, fixed_eps) 
                
                n_v = normal_vec(P[i])                
                dir_derv = w_grad @ n_v
                
                for j in range(num_stencil_nodes):
                    W_b[i,s[j]] = dir_derv[j]
                    
    return W_b

def boundary_to_weights_sparse(W, context, in_boundary, normal_vec, eps, fixed_eps=False):
    P = context.nodes
    S = context.stencils
    k = context.centers_info

    W_lil = W.tolil()

    for i,s in enumerate(S):
        node_type = in_boundary(P[i])

        if node_type != 'interior':
            if node_type == 'dirichlet':
                W_lil.rows[i] = [i]
                W_lil.data[i] = [1.0]
            elif node_type == 'neumann':
                if k is None:
                    w_grad = local_grad_solve(context, i, eps, fixed_eps)
                else:
                    w_grad = local_grad_ls(context, i, eps, fixed_eps)

                n_v = normal_vec(P[i])
                dir_derv = w_grad @ n_v

                W_lil.rows[i] = list(s)
                W_lil.data[i] = list(dir_derv)

    return W_lil.tocsr()

def right_hand_side(context, f, g, in_boundary):
    P = context.nodes
    A = context.A
    num_nodes = len(P)
    
    f_vec = np.zeros(num_nodes)
    
    for i,p in enumerate(P):
        node_type = in_boundary(P[i])
        if node_type == 'interior':            
            f_vec[i] = f(p,A[i])
        else:
            f_vec[i] = g(p)
        
    return f_vec

def is_pure_neumann(context, in_boundary):
    has_dirichlet = False
    has_boundary = False

    for p in context.nodes:
        lab = in_boundary(p)
        if lab != 'interior':
            has_boundary = True
        if lab == "dirichlet":
            has_dirichlet = True

    return has_boundary and not has_dirichlet

def anchor_system(W, f, method="mean"):
    W = W.copy()
    f = f.copy()

    n = W.shape[0]

    if method == "mean":
        # enforce sum(u)=0
        W[-1, :] = 1.0
        f[-1] = 0.0

    elif method == "pin":
        # fix one node
        W[0, :] = 0.0
        W[0, 0] = 1.0
        f[0] = 0.0

    elif method == "project":
        # projection method (more advanced, optional)
        e = np.ones(n)
        P = np.eye(n) - np.outer(e, e) / n
        W = P @ W @ P
        f = P @ f

    else:
        raise ValueError("Unknown anchoring method")

    return W, f

def anchor_system_sparse(W, f, method="mean"):
    f = f.copy()
    n = W.shape[0]
    W_lil = W.tolil()

    if method == "mean":
        W_lil.rows[-1] = list(range(n))
        W_lil.data[-1] = [1.0]*n
        f[-1] = 0.0

    elif method == "pin":
        W_lil.rows[0] = [0]
        W_lil.data[0] = [1.0]
        f[0] = 0.0

    elif method == "project":
        raise ValueError(
            "'project' anchoring is not supported for the sparse path: "
            "the projector is a dense rank-1 update and would silently "
            "densify the whole matrix. Use 'mean' or 'pin' instead, or "
            "use the dense anchor_system if 'project' is required."
        )

    else:
        raise ValueError("Unknown anchoring method")

    return W_lil.tocsr(), f

def set_rbf_func(basis, augmentation, context, tol=1e-12):
    context.set_augmentation(augmentation) 
 
    if (basis == 'gaussian'):
        context.set_phi(rbf.phi_gauss)
        context.set_grad_phi(rbf.grad_phi_gauss)
        context.set_laplacian_phi(rbf.anisotropic_diffusion_phi_gauss)        
    elif (basis == 'cubic'):
        context.set_phi(lambda p, eps: rbf.phi_cubic(p))
        context.set_grad_phi(lambda p, eps: rbf.grad_phi_cubic(p))
        context.set_laplacian_phi(
            lambda p, A, eps: rbf.anisotropic_diffusion_phi_cubic(p, A, tol))
        
    else:
        raise ValueError(f"'{basis}' is not a correct basis")

def set_boundary_func(g_bound, btype, shape, L, context):
    if (shape == 'square'):
        # Set g boundaries counterclockwise, starting with y=0
        in_boundary = lambda p: boundary.in_square_boundary(p, L, btype)
        normal_vec = lambda p: boundary.square_normal(p, L)
        g = lambda p: boundary.square_boundary(p, L, g_bound)
        
    elif (shape == 'circle'):
        in_boundary = lambda p: boundary.in_circular_boundary(p, L, btype)
        normal_vec = lambda p: boundary.disk_normal(p, L)
        g = lambda p: boundary.circular_boundary(p, L, g_bound)
        
    else:
        raise ValueError(f"'{shape}' is not a correct geometric boundary")
        
    return g, in_boundary, normal_vec

def rbf_fd_system(f, g_bound, btype, P, basis, shape, L, num_stencil_nodes,
                  num_centers_info, augmentation=False, A=None, eps=0.050,
                  fixed_eps=False, tol=1e-12, sparse=False, anchor_method="mean"):
    if A is None:
        A = np.ones((len(P[0]), 1, 1)) * np.eye(len(P[0]))

    S = stencils.knn_list(P, num_stencil_nodes)
    C = None
    
    if num_centers_info is not None:
        C = num_centers_info
    
    context = PDEDomainContext(P, S, C, A)    
    print('1) Nodes and Stencils Generated.')

    set_rbf_func(basis, augmentation, context, tol)
    g, in_boundary, normal_vec = set_boundary_func(g_bound, btype, shape, L, context)  
    print('2) RBF and Boundary information Stored.')
    
    if sparse:
        W = global_weights_sparse(context, eps, fixed_eps, in_boundary, normal_vec)
    else:
        W = global_weights(context, eps, fixed_eps, in_boundary, normal_vec)
    print('3) Weight Matrix Generated.')
            
    f_vec = right_hand_side(context, f, g, in_boundary)
    print('4) RHS vector Generated.')
    
    if is_pure_neumann(context, in_boundary):
        if sparse:
            W, f_vec = anchor_system_sparse(W, f_vec, method=anchor_method)
        else:
            W, f_vec = anchor_system(W, f_vec, method=anchor_method)
    print('5) Pure neumann condition checked and updated.')
    
    context.set_W(W)
    context.set_rhs(f_vec)
    print('6) Context Finished.')
    
    return context

def rbf_fd_solve(W, f_vec):
    return np.linalg.solve(W, f_vec)

def rbf_fd_solve_sparse(W, f_vec):
    return spla.spsolve(W.tocsr(), f_vec)

def coeff_matrix(nodes, eig_1, eig_2, angle):
    c = np.cos(angle(nodes))
    s = np.sin(angle(nodes))
    lambda1 = eig_1(nodes)
    lambda2 = eig_2(nodes)

    n = nodes.shape[1]
    A = np.zeros((n, 2, 2))
    
    A[:, 0, 0] = lambda1 * (c**2) + lambda2 * (s**2)      # Top-left (A_xx)
    A[:, 1, 1] = lambda1 * (s**2) + lambda2 * (c**2)      # Bottom-right (A_yy)
    
    off_diag = (lambda1 - lambda2) * c * s
    A[:, 0, 1] = off_diag                                 # Top-right (A_xy)
    A[:, 1, 0] = off_diag                                 # Bottom-left (A_yx)
    
    return A