# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:22:54 2026

@author: Dennis Corraliza

RBF-FD Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import geometry
import assembly_vec as assembly
import refinement
import error_analysis
import examples

def report_and_graph(context, u_exact, sparse=False):
    """
    Report solution-quality metrics and plot the RBF-FD solution, exact
    solution, and pointwise error.

    Solves the assembled linear system stored on `context` (via
    `assembly.rbf_fd_solve`), compares the result against the exact
    solution `u_exact` evaluated at the same nodes, and produces:

    - Console output reporting the condition number of the weight
      matrix `W`, the max/min entries of `W`, the maximum pointwise
      error, and the discrete L2 error.
    - A filled 2D contour plot of the approximate solution.
    - A 3D surface plot of the approximate solution.
    - A 3D surface plot of the exact solution.
    - A 3D surface plot of the pointwise absolute error.

    All plot titles annotate the anisotropic diffusion tensor entries
    A11 and A22.

    Parameters
    ----------
    context : PDEDomainContext
        Domain context providing the assembled weight matrix `context.W`
        and right-hand-side vector `context.F`.
    u_exact : callable
        Exact solution function, called as `u_exact(P.T)` to obtain the
        reference solution values at every node for comparison.

    Returns
    -------
    None
        Results are printed to stdout and figures are displayed via
        `plt.show()`; nothing is returned.
    """
    
    P = context.nodes
    A = context.A
    W = context.W
    F = context.F

    print('----------------------Solve Matrix Equation:----------------------')
    if sparse:
        u_soln = assembly.rbf_fd_solve_sparse(W, F)
    else:
        u_soln = assembly.rbf_fd_solve(W, F)
        print(f"Condition:    {np.linalg.cond(W): e}")

    print('------------------------Calculate Errors:-------------------------')
    u_ex = u_exact(P.T,np.transpose(A, (1, 2, 0)))
    error = np.abs(u_soln - u_ex)
    err_max = error_analysis.max_error_relative(u_soln, u_ex)
    err_l2 = error_analysis.l2_error_relative(u_soln, u_ex)
    #err_energy = error_analysis.energy_error_delaunay_relative(context, u_soln,
    #                                                           u_ex,sparse)
        
    Lu_approx = W @ u_ex
    res_max = error_analysis.max_error_relative(Lu_approx, F)
    res_l2 = error_analysis.l2_error_relative(Lu_approx, F)

    # -----------------------------
    # REPORT ERRORS AND PLOT
    # -----------------------------
    print('---------------------------Report:--------------------------------')
    X, Y = P.T
    
    # Provide error analysis from expected result
    print('Maximum weight:     ' + str(W.max()))
    print('Minimum weight:     ' + str(W.min()))
    print("Max error Rel     = ", np.max(err_max))
    print("L2 error Rel      = ", err_l2)
    #print("Energy error Rel  = ", err_energy)
    print("Res l2 Lu_ex - F  = ", res_l2)
    print("Res Max Lu_ex - F = ", res_max)

    print('--------------------------Graphics:--------------------------------')
    '''# Plot a contour for the approximated solution
    plt.figure(figsize=(8, 6))
    contour_filled = plt.tricontourf(X, Y, u_soln, levels=50, cmap='viridis')
    cbar = plt.colorbar(contour_filled)
    cbar.set_label('U Solution Value', rotation=270, labelpad=15)
    plt.title(rf"""RBF-FD Approximate Solution (Contour):""")
    plt.xlabel("x-direction")
    plt.ylabel("y-direction")
    #plt.show()
    
    # Plot a 3D graph of the approximated solution
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X,Y,u_soln,cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_zlim(u_ex.min(), u_ex.max())
    plt.title(rf"""RBF-FD Approximate Solution:""")
    
    plt.xlabel("x-direction")
    plt.ylabel("y-direction")
    #plt.show()
    
    # Plot a 3D graph of the exact solution
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X,Y,u_ex,cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_zlim(u_ex.min(), u_ex.max())
    plt.title(rf"""Exact Solution:""")
    plt.xlabel("x-direction")
    plt.ylabel("y-direction")
    #plt.show()

    # Plot a 3D graph of the error  
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X,Y,np.log10(np.maximum(error, 1e-16)),cmap='viridis', edgecolor='none', alpha=0.9)
    plt.title(rf"""RBF-FD Approximate Error:""")
    plt.xlabel("x-direction")
    plt.ylabel("y-direction")
    plt.show()'''

#%% Main Setup
if __name__ == "__main__":
    # -----------------------------
    # PARAMETERS
    # -----------------------------
    print('========================Define Parameters:========================')
    sparse = True
    
    # Define the nodes per stencil
    num_stencil_nodes = 20
    
    # Define the number of rings with quasi-uniform nodes
    # For Square solve, let num_centers := None
    num_centers = 15
    
    # Define the shape and parameters of the radial basis function
    rbf_shape = 'gaussian'
    augmentation = False

    # -----------------------------
    # BUILD NODES
    # -----------------------------
    Nx = 500
    Ny = 500
    L = 1.0
    shape = 'square'
    
    eps = 0.5*np.sqrt(Nx)
    tol = 1e-12

    print(f'Sparse Solve: {sparse}')
    print(f'Nx = {Nx}, Ny = {Ny}')
    print(f'Number of Stencils Nodes = {num_stencil_nodes}')
    print(f'Number of Rings          = {num_centers}')
    print(f'RBF: {rbf_shape} with augmentation: {augmentation}')
    print(f'Domain shape: {shape}')
    
    P, num_int = geometry.uniform_int_square(L, Nx, Ny,1)
    print("Node array shape: "+str(P.shape))
    
    # -----------------------------
    # ANISOTROPY AND PDE PROPERTIES
    # -----------------------------    
    # Define the conductivity condition
    eig_1 = lambda p: 1e0
    eig_2 = lambda p: 1e-3
    angle = lambda p: 12.0/24.0*np.pi
    A = assembly.coeff_matrix(P.T, eig_1, eig_2, angle)
    print("Coefficient Matrix shape:\n" + str(A.shape))
    
    # Forcing term parameters
    Amp = 1e3
    modes = [1.0,3.0]
    
    # -----------------------------
    # BUILD TEST CASE AND SOLVE
    # -----------------------------
    print('====================Define Example and Refine:====================')
    f, g, btype, u_exact = examples.example_10(eig_1, eig_2, angle, Amp, modes)
       
    '''P = refinement.mesh_refinement(f, g, btype, P, rbf_shape, shape, L,
                                   num_stencil_nodes, num_rings, augmentation,
                                   eig_1, eig_2, angle, eps, tol,
                                   sparse)'''

    print('Nodes Refined.')

    A = assembly.coeff_matrix(P.T, eig_1, eig_2, angle)
    print('Diffusion Tensor Redefined.')

    # Solve the PDE exactly and with RBF-FD
    print('========================Assemble System:==========================')
    context = assembly.rbf_fd_system(f, g, btype, P, rbf_shape, shape, L,
                                     num_stencil_nodes, num_centers, augmentation,
                                     A=A, eps=eps, tol=1e-8, sparse=sparse)
    
    # Display results
    print('===============Solve, Report and Graph Results:===================')
    report_and_graph(context, u_exact, sparse)