# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:04:22 2026

@author: Dennis Corraliza

Provides the assembly methods to solve FD Methods
"""

import numpy as np
import matplotlib.pyplot as plt
from domain import PDEDomainContext
import stencils
import rbf
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

    b = np.zeros(num_nodes)
    M = np.zeros((num_nodes, num_nodes))

    for k in range(num_nodes):
        b[k] = context.laplacian_phi(P[i]-P[s[k]])
        M[k,k] = context.phi(np.zeros(P[s[k]].shape))
        
        for j in range(k+1, num_nodes):
            M[j,k] = context.phi(P[s[j]]-P[s[k]])
            M[k,j] = M[j,k]
            
    if context.augmentation:
        Pmat = np.zeros((num_nodes, pdim))

        for j in range(num_nodes):
            Pmat[j, :] = rbf.poly_basis(P[s[j]])
             
        M = np.block([
            [M, Pmat],
            [Pmat.T, np.zeros((pdim,pdim))]
        ])
        
        b = np.concatenate([
            b,
            rbf.anisotropic_diffusion_poly(P[i], context.A)
        ])

    #print(f"Conditioning: {np.linalg.cond(M): e}")
    w = np.linalg.solve(M, b)
    return w[:num_nodes]

### Least Squares
def local_weights_ls(context, i):
    """
    Compute RBF-FD differentiation weights for node i via least squares.
 
    Alternative to `local_weights_solve` that avoids solving a square
    interpolation system directly. Instead, a cloud of auxiliary
    collocation centers is generated around node `i` (a local grid with
    `context.center_rings` rings, scaled by the stencil radius), and the
    weights are obtained as the least-squares solution that best
    reproduces the Laplacian of the kernel at those auxiliary centers
    using only the values at the stencil nodes. Polynomial augmentation
    constraints (if enabled) are enforced as exact equality rows rather
    than least-squares rows.
 
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
    
    P = context.nodes
    s = context.stencils[i]
    num_nodes = len(s)
    
    r = np.max(np.linalg.norm(P[s] - P[i], axis=1))
    k = context.center_rings

    points = rbf.generate_grid_2d(r, k)
    num_centers = len(points)
    c = P[i] + points

    # Augmentation form
    pdim = 0    
    if context.augmentation:
        pdim = len(rbf.poly_basis(P[0]))

    M = np.zeros((num_centers, num_nodes))
    b = np.zeros(num_centers)
    
    for k in range(num_centers):
        b[k] = context.laplacian_phi(P[i]-c[k])
        
        for j in range(num_nodes):
            M[k,j] = context.phi(P[s[j]]-c[k])
              
    if context.augmentation:        
        Pmat = np.zeros((num_nodes,   pdim))

        for j in range(num_nodes):
            Pmat[j, :] = rbf.poly_basis(P[s[j]])
                
        M = np.block([
            [M],
            [Pmat.T]
        ])
        
        b = np.concatenate([
            b,
            rbf.anisotropic_diffusion_poly(P[i], context.A)
        ])

    #print(f"Conditioning: {np.linalg.cond(M): e}")
    w = np.linalg.lstsq(M, b, rcond=None)[0]
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
    dim = len(P[0])

    # Augmentation form
    pdim = 0
    if context.augmentation:
        pdim = len(rbf.poly_basis(P[0]))

    b_grad = np.zeros((num_nodes,dim))    
    M = np.zeros((num_nodes, num_nodes))

    for k in range(num_nodes):
        b_grad[k,:] = context.grad_phi(P[i]-P[s[k]])        
        M[k,k] = context.phi(np.zeros(P[s[k]].shape))
        
        for j in range(k+1, num_nodes):
            M[j,k] = context.phi(P[s[j]]-P[s[k]])
            M[k,j] = M[j,k]
            
    if context.augmentation:
        Pmat = np.zeros((num_nodes, pdim))

        for j in range(num_nodes):
            Pmat[j, :] = rbf.poly_basis(P[s[j]])
       
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

def local_grad_ls(context, i):
    """
    Compute RBF-FD gradient weights for node i via least squares.
 
    Least-squares counterpart of `local_grad_solve`, following the same
    auxiliary-center strategy as `local_weights_ls`: a cloud of centers
    is generated around node `i`, and the gradient weights are the
    least-squares solution that best reproduces the gradient of the
    kernel (`context.grad_phi`) at those centers using only the stencil
    node values. Polynomial augmentation constraints (if enabled) are
    enforced as exact equality rows.
 
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
    num_nodes = len(s)
    
    P = context.nodes    
    dim = len(P[0])
    
    r = np.max(np.linalg.norm(P[s] - P[i], axis=1))
    k = context.center_rings
    
    points = rbf.generate_grid_2d(r, k)
    num_centers = len(points)
    c = P[i] + points

    # Augmentation form
    pdim = 0    
    if context.augmentation:
        pdim = len(rbf.poly_basis(P[0]))

    M = np.zeros((num_centers, num_nodes))
    b_grad = np.zeros((num_centers, dim))
    
    for k in range(num_centers):
        b_grad[k,:] = context.grad_phi(P[i]-c[k])
        
        for j in range(num_nodes):
            M[k,j] = context.phi(P[s[j]]-c[k])
              
    if context.augmentation:        
        Pmat = np.zeros((num_nodes, pdim))

        for j in range(num_nodes):
            Pmat[j, :] = rbf.poly_basis(P[s[j]])
                
        M = np.block([
            [M],
            [Pmat.T]
        ])
        
        b_grad = np.concatenate([
            b_grad,
            rbf.grad_poly(P[i])
        ])

    #print(f"Conditioning {np.linalg.cond(M): e}\n")   
    w_grad = np.linalg.lstsq(M, b_grad, rcond=None)[0]
    return w_grad[:num_nodes,:]

# .2 Assembles the global Weights.
def global_weights(context):
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
 
    Returns
    -------
    numpy.ndarray, shape (num_nodes, num_nodes)
        Dense global weight matrix discretizing the differential
        operator at every node, prior to applying boundary conditions.
    """

    P = context.nodes
    num_nodes = len(P)    
    
    S = context.stencils
    k = context.center_rings
    
    W = np.zeros((num_nodes, num_nodes))
    
    for i,s in enumerate(S):
        num_stencil_nodes = len(s)
        
        if k is None:
            w = local_weights_solve(context, i)
        else:
            w = local_weights_ls(context, i)

        for j in range(num_stencil_nodes):
            W[i,s[j]] = w[j]
            
    return W

def boundary_to_weights(W, context, in_boundary, normal_vec):
    """
    Overwrite global weight matrix rows for boundary nodes in place.
 
    For every node classified as non-interior by `is_boundary`, the
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
    is_boundary : callable
        Function mapping a node coordinate to a boundary-type label
        ('interior', 'dirichlet', 'neumann', ...).
    normal_vec : callable
        Function mapping a boundary node coordinate to its outward
        unit normal vector, used for Neumann rows.
 
    Returns
    -------
    None
        `W` is modified in place; nothing is returned.
    """

    P = context.nodes   
    
    S = context.stencils
    k = context.center_rings
    
    for i,s in enumerate(S):
        num_stencil_nodes = len(s)
        node_type = in_boundary(P[i])
        
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
    is_boundary : callable
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
    num_nodes = len(P)
    
    f_vec = np.zeros(num_nodes)
    
    for i,p in enumerate(P):
        node_type = in_boundary(P[i])
        if node_type == 'interior':            
            f_vec[i] = f(p)
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

def set_rbf_func(num_rings, basis, augmentation, A, eps, tol, context):
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
    A : numpy.ndarray or None
        Diffusion tensor used to build the anisotropic Laplacian
        operator for the chosen basis.
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
    
    if num_rings is not None:
        context.set_centers(num_rings)
        
    if (basis == 'gaussian'):
        context.set_phi(lambda p: rbf.phi_gauss(p, eps))
        context.set_grad_phi(lambda p: rbf.grad_phi_gauss(p, eps))
        context.set_laplacian_phi(
            lambda p: rbf.anisotropic_diffusion_phi_gauss(p, A, eps))
        
    elif (basis == 'cubic'):
        context.set_phi(rbf.phi_cubic)
        context.set_grad_phi(rbf.grad_phi_cubic)
        context.set_laplacian_phi(
            lambda p: rbf.anisotropic_diffusion_phi_cubic(p, A, tol))
        
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
                  num_rings, augmentation=False, A=None, eps=3.0, tol=1e-12):
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
 
    Returns
    -------
    PDEDomainContext
        Fully populated domain context, with `A`, the global weight
        matrix `W`, and the right-hand-side vector stored on it (via
        `set_A`, `set_W`, `set_rhs`), ready to be passed to
        `rbf_fd_solve`.
 
    Notes
    -----
    Progress messages are printed to stdout at each assembly stage.
    """
    
    if A is None:
        A = np.eye(len(P[0]))

    S = stencils.knn_list(P, num_stencil_nodes)
    context = PDEDomainContext(P, S, A)    
    print('1) Nodes and Stencils Generated.')

    set_rbf_func(num_rings, basis, augmentation, A, eps, tol, context)
    g, in_boundary, normal_vec = set_boundary_func(g_bound, btype, shape, L, context)  
    print('2) RBF and Boundary information Stored.')
    
    W = global_weights(context)
    print('3) Weight Matrix Generated.')
    boundary_to_weights(W, context, in_boundary, normal_vec)
    print('4) Boundary Conditions added to the weights.')
    
    f_vec = right_hand_side(context, f, g, in_boundary)
    print('5) RHS vector Generated.')
    
    if is_pure_neumann(context, in_boundary):
        W, f_vec = anchor_system(W, f_vec)
    print('6) Pure neumann condition checked and updated.')
    
    context.set_A(A)
    context.set_W(W)
    context.set_rhs(f_vec)
    print('9) Context Finished.')
    
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
    context = PDEDomainContext(P,S,np.eye(len(P[0])))

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