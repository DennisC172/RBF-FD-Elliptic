# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:39:40 2026

@author: Dennis Corraliza

Reconstruction Results for RBF-FD
"""

import numpy as np
import geometry
import assembly_vec as assembly
import examples
import error_analysis

def training_setup(L, Nx, Ny, shape, k_s, k_c, rbf_shape,
                   augmentation, eps, tol, alpha, beta,
                   angle_ratio, Amp, modes, example, sparse=False):
       
    P, num_int = geometry.uniform_int_square(L, Nx, Ny)
    
    # Define the conductivity condition
    theta = np.pi*angle_ratio
    D = np.array([[alpha, 0.0], [0.0, beta]])
    V = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),
                                                    np.cos(theta)]])
    
    A = V @ D @ V.T
    print("Coefficient Matrix:\n" + str(A))
    
    f, g_bound, btype, u_exact = example(Amp, modes, A) 
    
    # Solve the PDE exactly and with RBF-FD
    if sparse:
        context = assembly.rbf_fd_system(f, g_bound, btype, P,
                                         rbf_shape, shape, L, k_s, k_c,
                                         augmentation=augmentation,
                                         A=A, eps=eps, tol=tol, sparse=sparse)
    else:
        context = assembly.rbf_fd_system(f, g_bound, btype, P,
                                         rbf_shape, shape, L, k_s, k_c,
                                         augmentation=augmentation,
                                         A=A, eps=eps, tol=tol)        
    
    W = context.W
    F = context.F
    
    if sparse:
        u_train = assembly.rbf_fd_solve_sparse(W, F)
    else:
        u_train = assembly.rbf_fd_solve(W, F)
        print(f"Condition:    {np.linalg.cond(W): e}")
        
    u_ex = u_exact(P.T)
    err_max = error_analysis.max_error(u_train, u_ex)
    err_l2 = error_analysis.l2_error(u_train, u_ex)
    err_energy = error_analysis.energy_error_delaunay(context, u_train,
                                                      u_ex, sparse)
    
    print('Maximum weight:' + str(W.max()))
    print('Minimum weight:' + str(W.min()))
    print("Max error    = ", np.max(err_max))
    print("L2 error     = ", err_l2)
    print("Energy error = ", err_energy)
    
    return context

def reconstruction_analysis(context, shape, L, Amp, modes, example_problems, sparse=False):
    P = context.nodes
    A = context.A
    W = context.W
    
    for i, example in enumerate(example_problems):
        print('Example: ', str(i+3))
        f, g_bound, btype, u_exact = example(Amp, modes, A)    
        g, in_boundary, normal_vec = assembly.set_boundary_func(g_bound, btype,
                                                                shape, L,
                                                                context)
        if sparse:
            W = assembly.boundary_to_weights_sparse(W, context, in_boundary, normal_vec)
        else:
            W = assembly.boundary_to_weights(W, context, in_boundary, normal_vec)
            
        F = assembly.right_hand_side(context, f, g, in_boundary)
        u_ex = u_exact(P.T)
        Lu_approx = W @ u_ex
        
        idx = np.array([in_boundary(p) == 'interior' for p in P])  
        res_max    = error_analysis.max_error_relative(Lu_approx[idx], F[idx])
        res_l2     = error_analysis.l2_error_relative(Lu_approx[idx], F[idx])
        res_energy = error_analysis.energy_functional_delaunay_relative(context, u_ex,
                                                                        F, sparse)
        print("Max L_h u_exact-f Error Rel          = ", res_max)
        print("L2  L_h u_exact-f Error Rel          = ", res_l2)
        print("Energy Q(u_exact,u_exact)-F(u_exact) = ", res_energy)
        print()

if __name__ == "__main__":
    # -----------------------------
    # PARAMETERS
    # -----------------------------
    sparse = True
    
    # Define the nodes per stencil
    num_stencil_nodes = 100
    
    # Define the number of rings with quasi-uniform nodes
    # For Square solve, let k_c := None
    num_rings = 10
    
    # Define the shape and parameters of the radial basis function
    rbf_shape = 'cubic'
    augmentation = True
    eps = 3.0
    tol = 1e-12

    # -----------------------------
    # BUILD NODES
    # -----------------------------
    Nx = 50
    Ny = 50
    L = 1.0
    shape = 'square'
    
    # -----------------------------
    # ANISOTROPY AND PDE PROPERTIES
    # -----------------------------
    # Define the conductivity condition
    alpha = 1e0
    beta  = 5e-3
    angle_ratio = 12.0/24
    
    # Forcing term parameters
    Amp = 1.0
    modes = [1.0,3.0] 
       
    # Example list
    example_problems = [examples.example_3,
                        examples.example_4,
                        examples.example_5,
                        examples.example_6,
                        examples.example_7,
                        examples.example_8,
                        examples.example_9,
                        examples.example_10,
                       ]
    
    print('----------------------- Training Setup: --------------------------')    
    # -----------------------------
    # BUILD TRAIN CASE AND REPORT
    # -----------------------------
    context = training_setup(L, Nx, Ny, shape, num_stencil_nodes,
                             num_rings, rbf_shape, augmentation, eps,
                             tol, alpha, beta, angle_ratio,
                             Amp, modes, examples.example_2,sparse=sparse)
        
    # -----------------------------
    # RECONSTRUCT AND REPORT ERRORS
    # -----------------------------    
    print('---------------------- Testing Problems: -------------------------')
    reconstruction_analysis(context, shape, L, Amp, modes,
                            example_problems, sparse=sparse)