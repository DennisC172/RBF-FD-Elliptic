# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:39:40 2026

@author: Dennis Corraliza

Reconstruction Results for RBF-FD
"""

import numpy as np
import geometry
import assembly
import examples

def training_setup(L, Nx, Ny, shape, k_s, k_c, rbf_shape, augmentation, 
                   eps, tol, alpha, beta, angle, Amp, modes, example):
    
    f, g_bound, btype, u_exact = example()
    
    P, num_int = geometry.uniform_int_square(L, Nx, Ny)
    
    # Define the conductivity condition
    theta = np.pi*angle
    D = np.array([[alpha, 0.0], [0.0, beta]])
    V = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),
                                                    np.cos(theta)]])
    
    A = V @ D @ V.T
    print("Coefficient Matrix:\n" + str(A))
    
    # Solve the PDE exactly and with RBF-FD
    context = assembly.rbf_fd_system(f, g_bound, btype, P,
                                     rbf_shape, shape, L, k_s, k_c,
                                     augmentation=augmentation,
                                     A=A, eps=eps, tol=tol)
    
    W = context.W
    F = context.F
    u_train = assembly.rbf_fd_solve(W, F)
    u_ex = u_exact(P.T) 
    error = np.abs(u_train- u_ex)
  
    print(f"Condition:    {np.linalg.cond(W): e}")
    print('Maximum weight:' + str(W.max()))
    print('Minimum weight:' + str(W.min()))
    print("Max error =", np.max(error))
    print("L2 error  =", np.linalg.norm(error)/np.sqrt(len(P)))
    
    return context

def reconstruction_analysis(context, shape, L, Amp, modes, example_problems):
    P = context.nodes
    A = context.A
    W = context.W
    
    for i, example in enumerate(example_problems):
        print('Example: ', str(i+3))
        f, g_bound, btype, u_exact = example(Amp, modes, A)
    
        g, is_boundary, normal_vec = assembly.set_boundary_func(g_bound, btype,
                                                                shape, L,
                                                                context)
        assembly.boundary_to_weights(W, context, is_boundary, normal_vec)
        F = assembly.right_hand_side(context, f, g, is_boundary)
        u_ex = u_exact(P.T)
        Lu_approx = W @ u_ex
        
        idx = np.array([is_boundary(p) is not None for p in P])  
        error = np.abs((F[idx] - Lu_approx[idx]))
        print("Max error =", np.max(error))
        print("L2 error  =", np.linalg.norm(error)/np.sqrt(len(P)))
        print()

if __name__ == "__main__":
    # -----------------------------
    # PARAMETERS
    # -----------------------------
    
    # Define the nodes per stencil
    num_stencil_nodes= 50
    
    # Define the number of rings with quasi-uniform nodes
    # For Square solve, let k_c := None
    num_rings = 10
    
    # Define the shape and parameters of the radial basis function
    rbf_shape = 'gaussian'
    augmentation = True
    eps = 3.0
    tol = 1e-12

    # -----------------------------
    # BUILD NODES
    # -----------------------------
    Nx = 30
    Ny = 30
    L = 1.0    
    shape = 'square'
    
    # -----------------------------
    # ANISOTROPY AND PDE PROPERTIES
    # -----------------------------
    # Define the conductivity condition
    alpha = 1e0
    beta  = 1e-3
    angle = 10.15/24
    
    # Forcing term parameters
    Amp = 1.0
    modes = [7.0,3.0] 
       
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
                             tol, alpha, beta, angle,
                             Amp, modes, examples.example_2)
        
    # -----------------------------
    # RECONSTRUCT AND REPORT ERRORS
    # -----------------------------    
    print('---------------------- Testing Problems: -------------------------')
    reconstruction_analysis(context, shape, L, Amp, modes, example_problems)