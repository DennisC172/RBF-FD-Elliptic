# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:51:06 2026

@author: Dennis Corraliza

Error Analysis
"""

import numpy as np
import geometry
import assembly_vec as assembly
from scipy.spatial import Delaunay

def delaunay_lumped_weights(points, tri):
    """
    Lumped integration weights for a 2D point cloud.

    Parameters
    ----------
    points : (N,2) ndarray
    tri : scipy.spatial.Delaunay
        Delaunay triangulation.

    Returns
    -------
    weights : (N,) ndarray
        Nodal integration weights.
    """

    weights = np.zeros(len(points))

    for simplex in tri.simplices:
        p0, p1, p2 = points[simplex]

        area = 0.5 * abs(
            np.cross(p1 - p0, p2 - p0)
        )

        weights[simplex] += area / 3.0

    return weights

def l2_error(u_soln, u_ex):
    error = abs(u_soln-u_ex)
    return np.linalg.norm(error)/np.sqrt(len(error))

def l2_error_relative(u_soln, u_ex):
    error = abs(u_soln-u_ex)
    return np.linalg.norm(error)/np.linalg.norm(u_ex)   

def max_error(u_soln, u_ex):
    return np.max(abs(u_soln-u_ex))

def max_error_relative(u_soln, u_ex):
    return np.max(abs(u_soln-u_ex))/np.max(u_ex)

def energy_error(context, weights, u_soln, u_ex, sparse=False):
    P = context.nodes
    A = context.A
    num_nodes = len(P)
    dim = len(P[0])

    if sparse:
        W_grads = assembly.global_grads_sparse(context)
    else:
        W_grads = assembly.global_grads(context)
        
    err = u_soln-u_ex
    
    grad_err = np.zeros((num_nodes, dim))
    energy_err = 0.0
    
    for i in range(dim):
        if sparse:
            grad_err[:,i] = W_grads[i] @ err
        else:
            grad_err[:,i] = W_grads[:,:,i] @ err
    
    for i in range(num_nodes):
        energy_err += weights[i] * np.dot(grad_err[i,:], A @ grad_err[i,:])
    
    return np.sqrt(energy_err) 

def energy_error_delaunay(context, u_soln, u_ex, sparse=False):
    P = context.nodes
    tri = Delaunay(P)
    weights = delaunay_lumped_weights(P, tri)
    #print(weights.sum())
    
    return energy_error(context, weights, u_soln, u_ex, sparse)

def energy_error_delaunay_relative(context, u_soln, u_ex, sparse=False):
    P = context.nodes
    tri = Delaunay(P)
    weights = delaunay_lumped_weights(P, tri)
    #print(weights.sum())
    
    energy_err = energy_error(context, weights, u_soln, u_ex, sparse)
    dirichlet_energy = energy_error(context, weights, u_ex,
                                    np.zeros(u_ex.shape), sparse)
    
    return energy_err/dirichlet_energy

def energy_functional(context, weights, u_soln, f_vec, sparse=False):
    P = context.nodes
    A = context.A
    num_nodes = len(P)
    dim = len(P[0])

    if sparse:
        W_grads = assembly.global_grads_sparse(context)
    else:
        W_grads = assembly.global_grads(context)
    
    grad_u = np.zeros((num_nodes, dim))
    energy= 0.0
    
    for i in range(dim):
        if sparse:
            grad_u[:,i] = W_grads[i] @ u_soln
        else:
            grad_u[:,i] = W_grads[:,:,i] @ u_soln
    
    for i in range(num_nodes):
        energy += weights[i]*(np.dot(grad_u[i,:], A @ grad_u[i,:])/2
                              - f_vec[i] * u_soln[i])
    
    return energy

def energy_functional_delaunay(context, u_soln, f_vec, sparse=False):
    P = context.nodes
    tri = Delaunay(P)
    weights = delaunay_lumped_weights(P, tri)
    #print(weights.sum())
    
    return energy_functional(context, weights, u_soln, f_vec, sparse)

def coeff_matrix(eig_1, eig_2, rad24):
    """
    Build a symmetric 2x2 anisotropic diffusion tensor from eigenvalues
    and a rotation angle.

    Constructs a diagonal matrix `D = diag(eig1, eig2)` representing the
    diffusion coefficients along the tensor's principal axes, then
    rotates it by an angle `theta = pi * rad24 / 24` (i.e. `rad24` is
    expressed in 24ths of a half-turn, or 7.5-degree increments) via
    `A = V @ D @ V.T`, where `V` is the 2D rotation matrix for `theta`.
    The result is the diffusion tensor `A` expressed in the original
    (unrotated) x-y coordinate frame.

    Parameters
    ----------
    eig_1 : float
        Diffusion coefficient (eigenvalue) along the tensor's first
        principal axis.
    eig_2 : float
        Diffusion coefficient (eigenvalue) along the tensor's second
        principal axis.
    rad24 : float
        Rotation angle of the principal axes, in units of pi/24
        radians (e.g. `rad24 = 1` corresponds to a 7.5-degree
        rotation, `rad24 = 12` corresponds to 90 degrees).

    Returns
    -------
    numpy.ndarray, shape (2, 2)
        Symmetric positive (semi-)definite diffusion tensor `A`,
        rotated from the principal-axis frame into the standard x-y
        frame.
    """
    
    theta = np.pi*rad24/24.0
    D = np.array([[eig_1, 0.0], [0.0, eig_2]])
    V = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    return V @ D @ V.T  

def pde_context_provider(N_int, eig_2, rad24, example_problem):
    sparse = True
    
    # Define the nodes per stencil
    num_stencil_nodes = 150
    
    # Define the number of rings with quasi-uniform nodes
    # For Square solve, let k_c := None
    num_rings = 15
    
    # Define the shape and parameters of the radial basis function
    rbf_shape = 'cubic'
    augmentation = True
    eps = 3.0
    tol = 1e-12

    # -----------------------------
    # BUILD NODES
    # -----------------------------
    Nx = N_int
    Ny = N_int
    L = 1.0
    shape = 'square'
    
    P, num_int = geometry.uniform_int_square(L, Nx, Ny)
    
    # -----------------------------
    # ANISOTROPY AND PDE PROPERTIES
    # -----------------------------    
    # Define the conductivity condition
    eig_1 = 1e0
    A = coeff_matrix(eig_1, eig_2, rad24)
    print("Coefficient Matrix:\n" + str(A))
    
    # Forcing term parameters
    Amp = 1e3
    modes = [1.0,3.0]
    
    # -----------------------------
    # BUILD TEST CASE AND SOLVE
    # -----------------------------
    f, g, btype, u_exact = example_problem(Amp, modes, A)
    
    context = assembly.rbf_fd_system(f, g, btype, P,
                                 rbf_shape, shape, L, num_stencil_nodes,
                                 num_rings, augmentation=augmentation,
                                 A=A, eps=eps, tol=tol, sparse=sparse)
    
    P = context.nodes
    W = context.W
    F = context.F
    
    if sparse:
        u_soln = assembly.rbf_fd_solve_sparse(W, F)
    else:
        u_soln = assembly.rbf_fd_solve(W, F)
        print(f"Condition:    {np.linalg.cond(W): e}")
        
    u_ex = u_exact(P.T)
    error_l2 = l2_error(u_soln, u_ex)
    error_energy = energy_error_delaunay(context, u_soln, u_ex)
    
    return context, u_soln, u_ex, error_l2, error_energy