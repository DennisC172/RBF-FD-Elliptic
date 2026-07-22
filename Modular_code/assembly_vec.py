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

def local_weights_solve(context, i):
    """
    Compute RBF-FD differentiation weights for node i using direct collocation.
 
    Builds and solves the local RBF interpolation system (with optional
    polynomial augmentation) for the stencil of node `i`, returning the
    weights that approximate the Laplacian (or weighted Laplacian, as
    defined by `context.laplacian_phi`) at that node as a linear
    combination of the values at its stencil neighbors.
 
    The local system has the block form
 
        [ M   P ] [w]   [b]
        [ P^T 0 ] [l] = [0]
 
    where `M` is the RBF kernel (Gram) matrix evaluated at stencil node
    pairs, `P` holds polynomial basis evaluations (if augmentation is
    enabled), `b` holds the exact Laplacian of the kernel evaluated from
    the stencil center to each stencil node, and `w` are the desired
    weights.
 
    Parameters
    ----------
    context : PDEDomainContext
        Domain context providing `nodes`, `stencils`, `phi`,
        `laplacian_phi`, and `augmentation` settings.
    i : int
        Index of the node (in `context.nodes`) for which to compute
        local differentiation weights.
 
    Returns
    -------
    numpy.ndarray, shape (num_nodes,)
        RBF-FD weights for the Laplacian operator at node `i`, one
        weight per node in `context.stencils[i]`.
    """

    s = context.stencils[i]
    num_nodes = len(s)
    P = context.nodes

    # Augmentation form
    pdim = 0
    if context.augmentation:
        pdim = len(rbf.poly_basis(P[0]))

    # M: pairwise phi(P[s[j]] - P[s[k]]) for all (j,k), one vectorized call
    Ps = P[s]
    Ai = context.A[i]
    diff_M = Ps[:, None, :] - Ps[None, :, :]   # (num_nodes, num_nodes, dim)
    M = context.phi(diff_M)                     # (num_nodes, num_nodes)

    # b: laplacian_phi(P[i] - P[s[k]]) for all k, one vectorized call
    diff_b = P[i][None, :] - Ps                  # (num_nodes, dim)
    b = context.laplacian_phi(diff_b, Ai)             # (num_nodes,)

    if context.augmentation:
        Pmat = rbf.poly_basis(Ps)                 # (num_nodes, pdim)

        M = np.block([
            [M, Pmat],
            [Pmat.T, np.zeros((pdim,pdim))]
        ])
        
        b = np.concatenate([
            b,
            rbf.anisotropic_diffusion_poly(P[i])
        ])

    #print(f"Conditioning: {np.linalg.cond(M): e}")
    w = np.linalg.solve(M, b)
    return w[:num_nodes]

### Least Squares
def local_weights_ls(context, i, lam=0.0):
    """
    Compute RBF-FD differentiation weights for node i via least squares.
 
    Alternative to `local_weights_solve` that avoids solving a square
    interpolation system directly. Instead, a cloud of auxiliary
    collocation centers is generated around node `i` (a local grid with
    `context.center_rings` rings, scaled by the stencil radius), and the
    weights are obtained as the least-squares solution that best
    reproduces the Laplacian of the kernel at those auxiliary centers
    using only the values at the stencil nodes. Polynomial augmentation
    constraints (if enabled) are least-squares rows.
 
    Parameters
    ----------
    context : PDEDomainContext
        Domain context providing `nodes`, `stencils`, `phi`,
        `laplacian_phi`, `center_rings`, and `augmentation` settings.
    i : int
        Index of the node (in `context.nodes`) for which to compute
        local differentiation weights.
 
    Returns
    -------
    numpy.ndarray, shape (num_nodes,)
        Least-squares RBF-FD weights for the Laplacian operator at
        node `i`, one weight per node in `context.stencils[i]`.
    """

    s = context.stencils[i]
    c = context.centers[i]
    num_nodes = len(s)

    Ai = context.A[i]
    P = context.nodes    
    Ps = P[s] 
    Cs = P[s]

    #k = context.centers
    #r = np.max(np.linalg.norm(Ps - P[i], axis=1))
    #points = rbf.generate_grid_2d(r, k)
    #c = P[i] + points

    diff_M = Ps[None, :, :] - Cs[:, None, :]    # (num_centers, num_nodes, dim)
    M = context.phi(diff_M)                     # (num_centers, num_nodes)

    diff_b = P[i][None, :] - Cs                 # (num_centers, dim)
    b = context.laplacian_phi(diff_b, Ai)       # (num_centers, dim)

    if context.augmentation:
        Pmat = rbf.poly_basis(Ps)               # (num_nodes, pdim)
        M = np.block([
            [M],
            [Pmat.T]
        ])
        b = np.concatenate([
            b,
            rbf.grad_poly(P[i])                 # (pdim, dim)
        ])

    #print(f"Conditioning: {np.linalg.cond(M): e}")
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    filt = S / (S**2 + lam)
    w = Vt.T @ (filt * (U.T @ b))
    return w[:num_nodes]
    
def local_grad_solve(context, i):
    """
    Compute RBF-FD gradient weights for node i using direct collocation.
 
    Analogous to `local_weights_solve`, but builds weights that
    approximate the full gradient at node `i`
    as a linear combination of values at its stencil neighbors. Solves
    one augmented linear system whose right-hand side has one column
    per spatial dimension (`context.grad_phi`), so the returned weights
    can be dotted with a direction vector (e.g. an outward normal) to
    obtain directional-derivative weights.
 
    Parameters
    ----------
    context : PDEDomainContext
        Domain context providing `nodes`, `stencils`, `phi`,
        `grad_phi`, and `augmentation` settings.
    i : int
        Index of the node (in `context.nodes`) for which to compute
        local gradient weights.
 
    Returns
    -------
    numpy.ndarray, shape (num_nodes, dim)
        RBF-FD gradient weights at node `i`; column `d` gives the
        weights approximating the partial derivative with respect to
        spatial dimension `d`, one row per node in `context.stencils[i]`.
    """

    s = context.stencils[i]
    num_nodes = len(s)
    
    P = context.nodes

    # Augmentation form
    pdim = 0
    if context.augmentation:
        pdim = len(rbf.poly_basis(P[0]))

    Ps = P[s]
    # M: pairwise phi(P[s[j]] - P[s[k]]) for all (j,k), one vectorized call
    diff_M = Ps[:, None, :] - Ps[None, :, :]   # (num_nodes, num_nodes, dim)
    M = context.phi(diff_M)                     # (num_nodes, num_nodes)

    # b_grad: grad_phi(P[i] - P[s[k]]) for all k, one vectorized call
    diff_b = P[i][None, :] - Ps                  # (num_nodes, dim)
    b_grad = context.grad_phi(diff_b)             # (num_nodes, dim)

    if context.augmentation:
        Pmat = rbf.poly_basis(Ps)                 # (num_nodes, pdim)
       
        M = np.block([
            [M, Pmat],
            [Pmat.T, np.zeros((pdim,pdim))]
        ])
        
        b_grad = np.concatenate([
            b_grad,
            rbf.grad_poly(P[i])
        ])

    #print(f"Conditioning {np.linalg.cond(M): e}")
    w_grad = np.linalg.solve(M, b_grad)
    return w_grad[:num_nodes,:]

def local_grad_ls(context, i, lam=0.0):
    """
    Compute RBF-FD gradient weights for node i via least squares.
 
    Least-squares counterpart of `local_grad_solve`, following the same
    auxiliary-center strategy as `local_weights_ls`: a cloud of centers
    is generated around node `i`, and the gradient weights are the
    least-squares solution that best reproduces the gradient of the
    kernel (`context.grad_phi`) at those centers using only the stencil
    node values. Polynomial augmentation constraints (if enabled) are
    least-squares rows.
 
    Parameters
    ----------
    context : PDEDomainContext
        Domain context providing `nodes`, `stencils`, `phi`,
        `grad_phi`, `center_rings`, and `augmentation` settings.
    i : int
        Index of the node (in `context.nodes`) for which to compute
        local gradient weights.
 
    Returns
    -------
    numpy.ndarray, shape (num_nodes, dim)
        Least-squares RBF-FD gradient weights at node `i`; column `d`
        gives the weights approximating the partial derivative with
        respect to spatial dimension `d`, one row per node in
        `context.stencils[i]`.
    """
       
    s = context.stencils[i]
    c = context.centers[i]
    num_nodes = len(s)
    
    P = context.nodes    
    Ps = P[s] 
    Cs = P[c]

    #k = context.centers
    #r = np.max(np.linalg.norm(Ps - P[i], axis=1))
    #points = rbf.generate_grid_2d(r, k)
    #c = P[i] + points

    diff_M = Ps[None, :, :] - Cs[:, None, :]    # (num_centers, num_nodes, dim)
    M = context.phi(diff_M)                     # (num_centers, num_nodes)

    diff_b = P[i][None, :] - Cs                 # (num_centers, dim)
    b_grad = context.grad_phi(diff_b)           # (num_centers, dim)

    if context.augmentation:
        Pmat = rbf.poly_basis(Ps)               # (num_nodes, pdim)
        M = np.block([
            [M],
            [Pmat.T]
        ])
        b_grad = np.concatenate([
            b_grad,
            rbf.grad_poly(P[i])                 # (pdim, dim)
        ])

    #print(f"Conditioning: {np.linalg.cond(M): e}")
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    filt = S / (S**2 + lam)
    w = Vt.T @ (filt[:, None] * (U.T @ b_grad))
    return w[:num_nodes,:]

# .2 Assembles the global Weights.
def global_weights(context, in_boundary=None, normal_vec=None):
    """
    Assemble the global RBF-FD differentiation (Laplacian) matrix.
 
    Loops over every node in the domain, computes its local stencil
    weights (via `local_weights_solve` if `context.center_rings` is
    `None`, otherwise via `local_weights_ls`), and scatters them into
    the corresponding row/columns of a dense global matrix `W` such
    that `W @ u` approximates `-div(A grad u)` (or the operator encoded
    by `context.laplacian_phi`) at every node, ignoring boundary
    conditions.
 
    Parameters
    ----------
    context : PDEDomainContext
        Domain context providing `nodes`, `stencils`, and
        `center_rings`, used to dispatch to the direct or least-squares
        local weight routines.
    in_boundary : callable or None, optional
        Function mapping a node coordinate to a boundary-type label
        ('interior', 'dirichlet', 'neumann', ...). If provided (along
        with `normal_vec`), boundary nodes are assembled with their
        final Dirichlet/Neumann row directly -- the interior Laplacian
        stencil is never computed for those nodes in the first place
        (it would only be discarded or overwritten), so this avoids
        the wasted `local_weights_solve`/`local_weights_ls` call that
        `global_weights`/`boundary_to_weights` perform-then-discard
        for every boundary node. If `None` (the default), every node
        gets its interior Laplacian row regardless of boundary
        status, matching the original behavior -- callers can then
        apply boundary rows afterward via `boundary_to_weights_sparse`.
    normal_vec : callable or None, optional
        Function mapping a boundary node coordinate to its outward
        unit normal vector, used for Neumann rows. Required if
        `in_boundary` is provided; ignored (and may be left `None`) if
        `in_boundary` is `None`.
 
    Returns
    -------
    numpy.ndarray, shape (num_nodes, num_nodes)
        Dense global weight matrix discretizing the differential
        operator at every node, prior to applying boundary conditions.
        
    Notes
    -----
    Passing `in_boundary`/`normal_vec` here is an optional
    optimization, not a replacement for `boundary_to_weights_sparse`:
    if the same mesh/stencils will later be re-stamped with *different*
    boundary conditions (e.g. the reconstruction workflow in
    `reconstruction.py`, which reuses one assembled context across
    several example problems with different `btype`/`g_bound`), build
    `W` once with `in_boundary=None` here and call
    `boundary_to_weights_sparse` separately for each set of boundary
    conditions, rather than re-running this function from scratch each
    time.
    """

    P = context.nodes
    num_nodes = len(P)    
    
    S = context.stencils
    k = context.centers
    
    W = np.zeros((num_nodes, num_nodes))
    
    for i,s in enumerate(S):
        num_stencil_nodes = len(s)
        node_type = in_boundary(P[i]) if in_boundary is not None else 'interior'
        
        if node_type != 'interior':
            if node_type == 'dirichlet':
                W[i,:] = 0.0
                W[i,i] = 1.0
            elif node_type == 'neumann':
                if k is None:
                    w_grad = local_grad_solve(context, i)
                else:
                    w_grad = local_grad_ls(context, i) 
                
                n_v = normal_vec(P[i])                
                dir_derv = w_grad @ n_v
                
                for j in range(num_stencil_nodes):
                    W[i,s[j]] = dir_derv[j]
        else:
            if k is None:
                w = local_weights_solve(context, i)
            else:
                w = local_weights_ls(context, i)
    
            for j in range(num_stencil_nodes):
                W[i,s[j]] = w[j]
            
    return W

def global_weights_sparse(context, in_boundary=None, normal_vec=None):
    """
    Assemble the global RBF-FD differentiation (Laplacian) matrix in
    sparse (CSR) format, optionally applying boundary rows directly.

    Sparse counterpart of `global_weights`. Each row of the global
    weight matrix has only `len(context.stencils[i])` nonzero entries
    out of `num_nodes` columns -- for a typical stencil size (e.g. 50)
    on a mesh of several thousand nodes, the dense matrix is over 99%
    zeros. This function accumulates the per-node local weights as
    `global_weights` does, but scatters them directly into COO triplet
    arrays and builds a `scipy.sparse.csr_matrix` instead of a dense
    `numpy.ndarray`, avoiding both the `O(num_nodes^2)` memory
    footprint and the wasted "store/manipulate zeros" cost of the
    dense path.

    Parameters
    ----------
    context : PDEDomainContext
        Domain context providing `nodes`, `stencils`, and
        `center_rings`, used to dispatch to the direct or least-squares
        local weight routines.
    in_boundary : callable or None, optional
        Function mapping a node coordinate to a boundary-type label
        ('interior', 'dirichlet', 'neumann', ...). If provided (along
        with `normal_vec`), boundary nodes are assembled with their
        final Dirichlet/Neumann row directly -- the interior Laplacian
        stencil is never computed for those nodes in the first place
        (it would only be discarded or overwritten), so this avoids
        the wasted `local_weights_solve`/`local_weights_ls` call that
        `global_weights`/`boundary_to_weights` perform-then-discard
        for every boundary node. If `None` (the default), every node
        gets its interior Laplacian row regardless of boundary
        status, matching the original behavior -- callers can then
        apply boundary rows afterward via `boundary_to_weights_sparse`.
    normal_vec : callable or None, optional
        Function mapping a boundary node coordinate to its outward
        unit normal vector, used for Neumann rows. Required if
        `in_boundary` is provided; ignored (and may be left `None`) if
        `in_boundary` is `None`.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (num_nodes, num_nodes)
        Sparse global weight matrix. If `in_boundary`/`normal_vec`
        were provided, boundary rows already hold their final
        Dirichlet/Neumann form; otherwise every row holds the interior
        differential operator, prior to applying boundary conditions.

    Notes
    -----
    The local weight computation itself (`local_weights_solve` /
    `local_weights_ls` for interior rows, `local_grad_solve` /
    `local_grad_ls` for Neumann rows) is unchanged and still dense
    (each stencil's local system is small, e.g. 50x50, so a dense
    local solve is appropriate); only the *global* scatter-assembly
    step switches to a sparse representation. Converting from COO
    (good for incremental construction) to CSR (good for arithmetic
    and solves) is done once at the end via `tocsr()`, rather than
    repeatedly during assembly.

    Passing `in_boundary`/`normal_vec` here is an optional
    optimization, not a replacement for `boundary_to_weights_sparse`:
    if the same mesh/stencils will later be re-stamped with *different*
    boundary conditions (e.g. the reconstruction workflow in
    `reconstruction.py`, which reuses one assembled context across
    several example problems with different `btype`/`g_bound`), build
    `W` once with `in_boundary=None` here and call
    `boundary_to_weights_sparse` separately for each set of boundary
    conditions, rather than re-running this function from scratch each
    time.
    """

    P = context.nodes
    num_nodes = len(P)
    
    S = context.stencils
    k = context.centers
    
    max_nnz = num_nodes * context.stencils[0].shape[0]  # upper bound

    rows = np.empty(max_nnz, dtype=np.int32)
    cols = np.empty(max_nnz, dtype=np.int32)
    vals = np.empty(max_nnz, dtype=np.float64)
    ptr = 0
    
    for i,s in enumerate(S):
        node_type = in_boundary(P[i]) if in_boundary is not None else 'interior'

        if node_type == 'dirichlet':
            rows[ptr] = i
            cols[ptr] = i
            vals[ptr] = 1.0
            ptr += 1
        elif node_type == 'neumann':
            if k is None:
                w_grad = local_grad_solve(context, i)
            else:
                w_grad = local_grad_ls(context, i)

            n_v = normal_vec(P[i])
            dir_derv = w_grad @ n_v

            n = len(s)
            rows[ptr:ptr+n] = i
            cols[ptr:ptr+n] = s
            vals[ptr:ptr+n] = dir_derv
            ptr += n
        else:
            if k is None:
                w = local_weights_solve(context, i)
            else:
                w = local_weights_ls(context, i)

            n = len(s)
            rows[ptr:ptr+n] = i
            cols[ptr:ptr+n] = s
            vals[ptr:ptr+n] = w
            ptr += n
            
    W = sp.coo_matrix((vals[:ptr], (rows[:ptr], cols[:ptr])),
                      shape=(num_nodes, num_nodes))
    return W.tocsr()

def global_grads(context):
    """
    Assemble the global RBF-FD differentiation (Gradient) Tensor.
 
    Loops over every node in the domain, computes its local stencil
    weights (via `local_grads_solve` if `context.center_rings` is
    `None`, otherwise via `local_grads_ls`), and scatters them into
    the corresponding row/columns of a dense global matrix `W` such
    that `W @ u` approximates `grad u` (or the operator encoded
    by `context.grad_phi`) at every node
 
    Parameters
    ----------
    context : PDEDomainContext
        Domain context providing `nodes`, `stencils`, and
        `center_rings`, used to dispatch to the direct or least-squares
        local weight routines.
 
    Returns
    -------
    numpy.ndarray, shape (num_nodes, num_nodes, dim)
        Dense global weight matrix discretizing the gradient
        operator at every node.
    """

    P = context.nodes
    num_nodes = len(P)  
    dim = len(P[0])
    
    S = context.stencils
    k = context.center_rings
    
    W = np.zeros((num_nodes, num_nodes, dim))
    
    for i,s in enumerate(S):
        num_stencil_nodes = len(s)
        
        if k is None:
            w_grad = local_grad_solve(context, i)
        else:
            w_grad = local_grad_ls(context, i) 
            
        for j in range(num_stencil_nodes):
            for l in range(dim):
                W[i,s[j],l] = w_grad[j,l]
            
    return W

def global_grads_sparse(context):
    """
    Assemble the global RBF-FD differentiation (Gradient) operator in
    sparse (CSR) format, as a list of per-dimension matrices.

    Sparse counterpart of `global_grads`. Since `scipy.sparse` has no
    3D sparse tensor type, the dense `(num_nodes, num_nodes, dim)`
    tensor produced by `global_grads` is represented here as a list
    of `dim` separate `(num_nodes, num_nodes)` CSR matrices, one per
    spatial direction -- `W_list[l] @ u` approximates `d(u)/dx_l` at
    every node, matching `global_grads(context)[:, :, l] @ u`. Each
    row of `W_list[l]` has only `len(context.stencils[i])` nonzero
    entries out of `num_nodes` columns, so for typical stencil sizes
    on meshes of several thousand nodes this avoids the same
    `O(num_nodes^2)` memory cost that `global_weights_sparse` avoids
    for the Laplacian.

    Parameters
    ----------
    context : PDEDomainContext
        Domain context providing `nodes`, `stencils`, and
        `center_rings`, used to dispatch to the direct or
        least-squares local gradient-weight routines.

    Returns
    -------
    list of scipy.sparse.csr_matrix, length dim
        `W_list[l]` is the sparse global weight matrix discretizing
        `d/dx_l` at every node, for `l = 0, ..., dim-1`.

    Notes
    -----
    The local weight computation (`local_grad_solve` / `local_grad_ls`)
    is unchanged and still dense -- only the global scatter-assembly
    switches to sparse. COO triplet lists are accumulated per
    dimension during the loop and each converted to CSR once at the
    end via `tocsr()`, rather than repeatedly during assembly.
    """

    P = context.nodes
    num_nodes = len(P)
    dim = len(P[0])

    S = context.stencils
    k = context.center_rings

    max_stencil = max(len(s) for s in S)
    max_nnz = num_nodes * max_stencil

    rows = np.empty(max_nnz, dtype=np.int64)
    cols = np.empty(max_nnz, dtype=np.int64)
    vals = np.empty((dim, max_nnz), dtype=np.float64)
    ptr = 0

    for i, s in enumerate(S):
        n = len(s)

        if k is None:
            w_grad = local_grad_solve(context, i)
        else:
            w_grad = local_grad_ls(context, i)

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

def boundary_to_weights(W, context, in_boundary, normal_vec):
    """
    Overwrite global weight matrix rows for boundary nodes in place.
 
    For every node classified as non-interior by `in_boundary`, the
    corresponding row of `W` is replaced:
 
    - Dirichlet nodes: the row is zeroed out and set to the identity
      (`W[i, i] = 1`), so that the solution at that node is pinned
      directly to the boundary value supplied in the right-hand side.
    - Neumann nodes: the row is replaced with directional-derivative
      weights, computed from the local gradient weights
      (`local_grad_solve` or `local_grad_ls`, depending on
      `context.center_rings`) dotted with the outward normal vector at
      that node, so that the row enforces `du/dn = g` at that node.
 
    Interior nodes are left untouched.
 
    Parameters
    ----------
    W : numpy.ndarray, shape (num_nodes, num_nodes)
        Global weight matrix (as produced by `global_weights`),
        modified in place.
    context : PDEDomainContext
        Domain context providing `nodes`, `stencils`, and
        `center_rings`.
    in_boundary : callable
        Function mapping a node coordinate to a boundary-type label
        ('interior', 'dirichlet', 'neumann', ...).
    normal_vec : callable
        Function mapping a boundary node coordinate to its outward
        unit normal vector, used for Neumann rows.
 
    Returns
    -------
    numpy.ndarray, shape (num_nodes,num_nodes)
        New sparse matrix with boundary rows imposed.
    """

    P = context.nodes   
    
    S = context.stencils
    k = context.center_rings
    
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
                    w_grad = local_grad_solve(context, i)
                else:
                    w_grad = local_grad_ls(context, i) 
                
                n_v = normal_vec(P[i])                
                dir_derv = w_grad @ n_v
                
                for j in range(num_stencil_nodes):
                    W_b[i,s[j]] = dir_derv[j]
                    
    return W_b

def boundary_to_weights_sparse(W, context, in_boundary, normal_vec):
    """
    Overwrite global sparse weight matrix rows for boundary nodes.

    Sparse counterpart of `boundary_to_weights`. Row-by-row
    replacement on a `csr_matrix` is expensive (each assignment can
    trigger an internal re-sort/re-index of that row's nonzero
    pattern), so this function first converts `W` to `lil_matrix`
    format (which supports efficient row-wise mutation), performs the
    same Dirichlet/Neumann row replacements as `boundary_to_weights`,
    and converts the result back to `csr_matrix` for the downstream
    sparse solve.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (num_nodes, num_nodes)
        Global sparse weight matrix (as produced by
        `global_weights_sparse`). Not modified in place -- a
        `lil_matrix` copy is mutated internally and a new `csr_matrix`
        is returned, mirroring `anchor_system`'s copy-rather-than-
        mutate convention (unlike the dense `boundary_to_weights`,
        which does mutate `W` in place; this asymmetry exists because
        in-place row replacement is the efficient option for a dense
        `numpy.ndarray` but not for a `csr_matrix`).
    context : PDEDomainContext
        Domain context providing `nodes`, `stencils`, and
        `center_rings`.
    in_boundary : callable
        Function mapping a node coordinate to a boundary-type label
        ('interior', 'dirichlet', 'neumann', ...).
    normal_vec : callable
        Function mapping a boundary node coordinate to its outward
        unit normal vector, used for Neumann rows.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (num_nodes, num_nodes)
        New sparse matrix with boundary rows imposed.
    """

    P = context.nodes
    S = context.stencils
    k = context.center_rings

    W_lil = W.tolil()

    for i,s in enumerate(S):
        node_type = in_boundary(P[i])

        if node_type != 'interior':
            if node_type == 'dirichlet':
                W_lil.rows[i] = [i]
                W_lil.data[i] = [1.0]
            elif node_type == 'neumann':
                if k is None:
                    w_grad = local_grad_solve(context, i)
                else:
                    w_grad = local_grad_ls(context, i)

                n_v = normal_vec(P[i])
                dir_derv = w_grad @ n_v

                W_lil.rows[i] = list(s)
                W_lil.data[i] = list(dir_derv)

    return W_lil.tocsr()

def right_hand_side(context, f, g, in_boundary):
    """
    Assemble the global right-hand-side vector for the linear system.
 
    For each node, the entry is set to the source term `f(p)` if the
    node is interior, or to the boundary data `g(p)` if the node lies
    on the boundary (Dirichlet value or Neumann flux, depending on how
    `g` and the corresponding row of `W` were constructed).
 
    Parameters
    ----------
    context : PDEDomainContext
        Domain context providing `nodes`.
    f : callable
        Source/forcing function, called as `f(p)` for interior node
        coordinates `p`.
    g : callable
        Boundary data function, called as `g(p)` for boundary node
        coordinates `p`.
    i_boundary : callable
        Function mapping a node coordinate to a boundary-type label;
        used only to decide whether `f` or `g` is evaluated at each
        node.
 
    Returns
    -------
    numpy.ndarray, shape (num_nodes,)
        Right-hand-side vector matching the row ordering of the global
        weight matrix `W`.
    """

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
    """
    Check whether the domain has boundary nodes but no Dirichlet nodes.
 
    A pure Neumann problem (Laplace/Poisson-type with flux-only boundary
    conditions everywhere) determines its solution only up to a
    constant, so the resulting linear system is singular and
    requires anchoring (see `anchor_system`) before it can be solved.
    This function flags that situation.
 
    Parameters
    ----------
    context : PDEDomainContext
        Domain context providing `nodes`.
    is_boundary : callable
        Function mapping a node coordinate to a boundary-type label
        ('interior', 'dirichlet', 'neumann', ...).
 
    Returns
    -------
    bool
        `True` if the domain has at least one boundary node and no
        node is labeled `'dirichlet'`; `False` otherwise (including the
        case of a fully interior/unbounded domain with no boundary
        nodes at all).
    """

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
    """
    Regularize a singular pure-Neumann system so it has a unique solution.
 
    Pure Neumann systems are singular because the solution is only
    determined up to a constant. This function modifies a
    copy of `W` and `f` to remove that null space, using one of three
    strategies.
 
    Parameters
    ----------
    W : numpy.ndarray, shape (n, n)
        Global weight matrix to anchor. Not modified in place; a copy
        is used internally.
    f : numpy.ndarray, shape (n,)
        Right-hand-side vector to anchor. Not modified in place; a copy
        is used internally.
    method : {'mean', 'pin', 'project'}, optional
        Anchoring strategy to apply (default `'mean'`):
 
        - `'mean'`: replace the last row of `W` with all ones and the
          last entry of `f` with 0, enforcing `sum(u) = 0`.
        - `'pin'`: replace the first row of `W` with the identity row
          and set `f[0] = 0`, pinning the solution at node 0 to zero.
        - `'project'`: project both `W` and `f` onto the subspace
          orthogonal to the constant vector, using the projector
          `P = I - (1/n) * ones @ ones.T`.
 
    Returns
    -------
    W : numpy.ndarray, shape (n, n)
        The anchored weight matrix.
    f : numpy.ndarray, shape (n,)
        The anchored right-hand-side vector.
 
    Raises
    ------
    ValueError
        If `method` is not one of `'mean'`, `'pin'`, or `'project'`.
    """

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
    """
    Regularize a singular pure-Neumann sparse system so it has a
    unique solution.

    Sparse counterpart of `anchor_system`, supporting the `'mean'` and
    `'pin'` strategies (see `anchor_system` for their definitions).
    `'project'` is intentionally not supported here: the projector
    `P = I - (1/n) * ones @ ones.T` is a dense rank-1 update, and
    `P @ W @ P` is generically a fully dense matrix even when `W` is
    sparse, since the projector mixes every row and column together.
    Silently densifying defeats the purpose of the sparse path, so
    `'project'` raises rather than falling back to a dense computation
    the caller didn't ask for.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Global sparse weight matrix to anchor. Not modified in place;
        a copy is used internally.
    f : numpy.ndarray, shape (n,)
        Right-hand-side vector to anchor. Not modified in place; a
        copy is used internally.
    method : {'mean', 'pin'}, optional
        Anchoring strategy to apply (default `'mean'`). See
        `anchor_system` for the definitions of `'mean'` and `'pin'`;
        both are implemented identically here, just via sparse row
        assignment instead of dense.

    Returns
    -------
    W : scipy.sparse.csr_matrix, shape (n, n)
        The anchored sparse weight matrix.
    f : numpy.ndarray, shape (n,)
        The anchored right-hand-side vector.

    Raises
    ------
    ValueError
        If `method` is `'project'` (unsupported for the sparse path;
        see Notes) or anything other than `'mean'`/`'pin'`.
    """

    f = f.copy()
    n = W.shape[0]
    W_lil = W.tolil()

    if method == "mean":
        # enforce sum(u)=0 -- note this row is fully dense (all ones),
        # so the resulting matrix is no longer sparse in that one row,
        # but this is unavoidable for this anchoring strategy and is
        # still far cheaper than densifying the entire matrix.
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

def set_rbf_func(basis, augmentation, eps, tol, context):
    """
    Configure a domain context with the chosen RBF kernel and operator.
 
    Sets the polynomial-augmentation flag, optionally the number of
    auxiliary center rings (for least-squares stencils), and the three
    kernel callables (`phi`, `grad_phi`, `laplacian_phi`) on `context`
    according to the requested basis function. The Laplacian callable
    is built to represent the (possibly anisotropic) operator
    `div(A grad phi)`, using the diffusion tensor `A`.
 
    Parameters
    ----------
    num_rings : int or None
        Number of auxiliary center rings to use for least-squares
        stencil weights. If not `None`, stored on `context` via
        `context.set_centers`.
    basis : {'gaussian', 'cubic'}
        Name of the radial basis function to use.
    augmentation : bool
        Whether to augment the RBF interpolant with a polynomial basis
    eps : float
        Shape parameter for the Gaussian RBF (used only when
        `basis == 'gaussian'`).
    tol : float
        Tolerance parameter passed to the cubic RBF anisotropic
        Laplacian (used only when `basis == 'cubic'`).
    context : PDEDomainContext
        Domain context to configure in place.
 
    Returns
    -------
    None
        `context` is modified in place; nothing is returned.
 
    Raises
    ------
    ValueError
        If `basis` is not `'gaussian'` or `'cubic'`.
    """

    context.set_augmentation(augmentation) 
        
    if (basis == 'gaussian'):
        context.set_phi(lambda p: rbf.phi_gauss(p, eps))
        context.set_grad_phi(lambda p: rbf.grad_phi_gauss(p, eps))
        context.set_laplacian_phi(
            lambda p, A: rbf.anisotropic_diffusion_phi_gauss(p, A, eps))
        
    elif (basis == 'cubic'):
        context.set_phi(rbf.phi_cubic)
        context.set_grad_phi(rbf.grad_phi_cubic)
        context.set_laplacian_phi(
            lambda p, A: rbf.anisotropic_diffusion_phi_cubic(p, A, tol))
        
    else:
        raise ValueError(f"'{basis}' is not a correct basis")

def set_boundary_func(g_bound, btype, shape, L, context):
    """
    Build the boundary-data, classification, and normal-vector callables.
 
    Dispatches on `shape` to construct three closures tailored to the
    requested domain geometry, wiring in the boundary type labels
    (`btype`) and boundary-value functions (`g_bound`) from
    `boundary.square_boundary` / `boundary.circular_boundary` and the
    matching classification/normal routines from the `boundary` module.
 
    Parameters
    ----------
    g_bound : list of callables or None
        Boundary value functions for each side/arc of the domain,
        passed through to the boundary-evaluation closure.
    btype : list of str
        Boundary condition type for each side/arc of the domain (e.g.
        `'dirichlet'`, `'neumann'`), passed through to the
        classification closure.
    shape : {'square', 'circle'}
        Domain geometry.
    L : float
        Characteristic size of the domain (side length for `'square'`,
        radius for `'circle'`).
    context : PDEDomainContext
        Domain context (accepted for interface consistency; not
        directly used to build the closures).
 
    Returns
    -------
    g : callable
        Function mapping a boundary node coordinate to its prescribed
        boundary value/flux.
    in_boundary : callable
        Function mapping a node coordinate to its boundary-type label
        (e.g. `'interior'`, `'dirichlet'`, `'neumann'`).
    normal_vec : callable
        Function mapping a boundary node coordinate to its outward
        unit normal vector.
 
    Raises
    ------
    ValueError
        If `shape` is not `'square'` or `'circle'`.
    """

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
                  num_center_nodes, augmentation=False, A=None, eps=3.0, tol=1e-12,
                  sparse=False, anchor_method="mean"):
    """
    Build the full RBF-FD linear system for a (possibly anisotropic) PDE.
 
    This is the top-level assembly routine: it builds the domain
    context and stencils, configures the RBF kernel and boundary
    handling, assembles the global weight matrix, imposes the boundary
    conditions, builds the right-hand side, anchors the system if it is
    pure Neumann, and stores everything on the returned context for
    later solution.
 
    Parameters
    ----------
    f : callable
        Source/forcing function for interior nodes, called as `f(p)`.
    g_bound : list of callables
        Boundary value functions for each side/arc of the domain.
    btype : list of str
        Boundary condition type for each side/arc of the domain (e.g.
        `'dirichlet'`, `'neumann'`).
    P : numpy.ndarray, shape (num_nodes, dim)
        Node coordinates discretizing the domain.
    basis : {'gaussian', 'cubic'}
        Radial basis function to use for the local RBF-FD stencils.
    shape : {'square', 'circle'}
        Domain geometry.
    L : float
        Characteristic size of the domain (side length or radius).
    num_stencil_nodes : int
        Number of nearest neighbors used to build each node's stencil.
    num_rings : int or None
        Number of auxiliary center rings for least-squares stencil
        weights; if `None`, direct collocation is used instead.
    augmentation : bool, optional
        Whether to augment RBF interpolants with a polynomial basis
        (default `False`).
    A : numpy.ndarray or None, optional
        Diffusion tensor for the (anisotropic) Laplacian operator
        (default `None`).
    eps : float, optional
        Shape parameter for the Gaussian RBF (default `3.0`).
    tol : float, optional
        Tolerance parameter for the cubic RBF anisotropic Laplacian
        (default `1e-12`).
    sparse : bool, optional
        If `True`, assemble and store `context.W` as a
        `scipy.sparse.csr_matrix` (via `global_weights_sparse` and
        `boundary_to_weights_sparse`) instead of a dense
        `numpy.ndarray`. Use `rbf_fd_solve_sparse` (not
        `rbf_fd_solve`) to solve the resulting system. Default `False`
        preserves the original dense behavior for existing callers.
    anchor_method : str, optional
        Anchoring strategy passed to `anchor_system`
        (`sparse=False`) or `anchor_system_sparse` (`sparse=True`) if
        the problem is pure Neumann (default `"mean"`). Note
        `anchor_system_sparse` does not support `"project"` (see its
        docstring); pass `sparse=False` if that method is required.
 
    Returns
    -------
    PDEDomainContext
        Fully populated domain context, with `A`, the global weight
        matrix `W` (dense or sparse depending on `sparse`), and the
        right-hand-side vector stored on it (via `set_A`, `set_W`,
        `set_rhs`), ready to be passed to `rbf_fd_solve` or
        `rbf_fd_solve_sparse` accordingly.
 
    Notes
    -----
    Progress messages are printed to stdout at each assembly stage.
    """
    
    if A is None:
        A = np.ones((len(P[0]), 1, 1)) * np.eye(len(P[0]))

    S = stencils.knn_list(P, num_stencil_nodes)
    C = stencils.knn_list(P, num_center_nodes)
    context = PDEDomainContext(P, S, C, A)    
    print('1) Nodes and Stencils Generated.')

    set_rbf_func(basis, augmentation, eps, tol, context)
    g, in_boundary, normal_vec = set_boundary_func(g_bound, btype, shape, L, context)  
    print('2) RBF and Boundary information Stored.')
    
    if sparse:
        W = global_weights_sparse(context, in_boundary, normal_vec)
    else:
        W = global_weights(context, in_boundary, normal_vec)
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
    """
    Solve the assembled RBF-FD linear system for the nodal solution.
 
    Thin wrapper around `numpy.linalg.solve` that solves `W @ u = f_vec`
    for the unknown nodal values `u`.
 
    Parameters
    ----------
    W : numpy.ndarray, shape (n, n)
        Global weight matrix (with boundary conditions imposed, and
        anchored if pure Neumann), as produced by `rbf_fd_system`.
    f_vec : numpy.ndarray, shape (n,)
        Right-hand-side vector corresponding to `W`.
 
    Returns
    -------
    numpy.ndarray, shape (n,)
        Nodal solution vector `u` satisfying `W @ u = f_vec`.
    """

    return np.linalg.solve(W, f_vec)

def rbf_fd_solve_sparse(W, f_vec):
    """
    Solve the assembled sparse RBF-FD linear system for the nodal
    solution.

    Sparse counterpart of `rbf_fd_solve`, using
    `scipy.sparse.linalg.spsolve` (a sparse LU factorization) instead
    of `numpy.linalg.solve`. For a problem with `n` nodes and stencils
    of size `k << n`, `W` has only `O(n*k)` nonzero entries rather than
    `O(n^2)`; a sparse direct solve exploits this structure to run in
    far less time and memory than the dense path, particularly as `n`
    grows (the dense path's `numpy.linalg.solve` is `O(n^3)` and its
    memory is `O(n^2)` regardless of how sparse the underlying physics
    is, whereas a sparse LU factorization's cost scales with the
    matrix's actual nonzero/fill-in structure).

    Parameters
    ----------
    W : scipy.sparse.csr_matrix (or other scipy.sparse format), shape (n, n)
        Global sparse weight matrix (with boundary conditions imposed,
        and anchored if pure Neumann), as produced by `rbf_fd_system`
        with `sparse=True`.
    f_vec : numpy.ndarray, shape (n,)
        Right-hand-side vector corresponding to `W`.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Nodal solution vector `u` satisfying `W @ u = f_vec`.

    Notes
    -----
    `spsolve` expects CSC or CSR input; if `W` is some other sparse
    format (e.g. the `lil_matrix` intermediate used internally by
    `boundary_to_weights_sparse`/`anchor_system_sparse` before their
    final `.tocsr()` conversion), convert it to CSR first -- the
    functions in this module already return CSR, so this is only a
    concern for callers building `W` by other means.
    """

    return spla.spsolve(W.tocsr(), f_vec)

def coeff_matrix(nodes, eig_1, eig_2, angle):
    """
    Build a symmetric 2x2 anisotropic diffusion tensor from eigenvalues
    and a rotation angle.

    Constructs a diagonal matrix `D = diag(eig1, eig2)` representing the
    diffusion coefficients along the tensor's principal axes, then
    rotates it by an 'angle' via `A = V @ D @ V.T`, where `V` is the
    2D rotation matrix for `angle`.
    The result is the diffusion tensor `A` expressed in the original
    (unrotated) x-y coordinate frame.

    Parameters
    ----------
    nodes : array_like, shape (n, 2)
        Spatial locations at which to evaluate the diffusion tensor.
    eig_1 : callable
        Function of `nodes` returning the diffusion coefficient
        (eigenvalue) along the tensor's first principal axis, shape (n,).
    eig_2 : callable
        Function of `nodes` returning the diffusion coefficient
        (eigenvalue) along the tensor's second principal axis, shape (n,).
    angle : callable
        Function of `nodes` returning the rotation angle of the
        principal axes, shape (n,).

    Returns
    -------
    numpy.ndarray, shape (n , 2, 2)
        Symmetric positive (semi-)definite diffusion tensor `A`,
        rotated from the principal-axis frame into the standard x-y
        frame.
    """
    
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

if __name__ == "__main__":
    import geometry
    # -----------------------------
    # PARAMETERS
    # -----------------------------
    num_stencil_nodes = 25
    num_rings = 10

    rbf_shape = 'cubic'
    augmentation = True
    eps = 3.0
    tol = 1e-12

    Nx = 50
    Ny = 50
    L = 1.0

    shape = 'square'

    # -----------------------------
    # BUILD NODES
    # -----------------------------
    P, num_int = geometry.uniform_int_square(L, Nx, Ny)

    # -----------------------------
    # BUILD CONTEXT
    # -----------------------------
    S = stencils.knn_list(P, num_stencil_nodes)
    context = PDEDomainContext(P,S,np.ones((len(P[0]), 1, 1))
                               *np.eye(len(P[0])))

    set_rbf_func(
        num_rings=num_rings,
        basis=rbf_shape,
        augmentation=augmentation,
        A=None,
        eps=eps,
        tol=tol,
        context=context
    )

    # boundary functions
    g, is_boundary, normal_vec = set_boundary_func(
        g_bound=None,
        btype=["neumann", "neumann", "neumann", "neumann"],
        shape=shape,
        L=L,
        context=context
    )

    # -----------------------------
    # FIND NEUMANN NODES
    # -----------------------------
    labels = [is_boundary(P[i]) == 'neumann' for i in range(len(P))]

    print("Unique boundary labels:", set(labels))
    print("Counts:",
          {lab: labels.count(lab) for lab in set(labels)})
    
    neumann_nodes = [
        i for i in range(len(P))
        if is_boundary(P[i]) == "neumann"
    ]

    print(f"Neumann nodes: {len(neumann_nodes)}")

    # -----------------------------
    # ERROR STORAGE
    # -----------------------------
    err_const = []
    err_x = []
    err_y = []

    # -----------------------------
    # TEST LOOP
    # -----------------------------
    for i in neumann_nodes:
        s = context.stencils[i]
        w_grad = local_grad_ls(context, i)

        n = normal_vec(P[i])
        dir_derv = w_grad @ n

        # stencil values
        ones = np.ones(len(s))
        x_s = P[s, 0]
        y_s = P[s, 1]

        Wx = w_grad[:,0]
        Wy = w_grad[:,1]

        # -------------------------
        # REPRODUCTION TESTS
        # -------------------------
        err_const.append(abs(dir_derv @ ones))
        err_x.append(abs(dir_derv @ x_s - n[0]))
        err_y.append(abs(dir_derv @ y_s - n[1]))           

    # -----------------------------
    # REPORT
    # -----------------------------
    print("Neumann reproduction errors:")
    
    if len(err_const) == 0:
        print("No Neumann nodes detected!")
    else:
        print("max error: ", np.max(err_const))

    print(f"Max x-error:        {np.max(err_x):.3e}")
    print(f"Max y-error:        {np.max(err_y):.3e}")
       
    plt.scatter(
        P[neumann_nodes,0],
        P[neumann_nodes,1],
        c=err_const,
        s=15,
        cmap='viridis'
    )
    
    plt.colorbar(label='Reproduction Error')
    plt.axis('equal')
    plt.show()
    
    plt.scatter(
        P[neumann_nodes,0],
        P[neumann_nodes,1],
        c=err_x,
        s=15,
        cmap='viridis'
    )
    
    plt.colorbar(label='Reproduction Error')
    plt.axis('equal')    
    plt.show()
    
    plt.scatter(
        P[neumann_nodes,0],
        P[neumann_nodes,1],
        c=err_y,
        s=15,
        cmap='viridis'
    )

    plt.colorbar(label='Reproduction Error')
    plt.axis('equal')    
    plt.show()