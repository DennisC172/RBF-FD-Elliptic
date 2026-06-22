# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:22:54 2026

@author: Dennis Corraliza

RBF-FD Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import geometry
import assembly
import examples

def report_and_graph(context, u_exact):
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
    u_soln = assembly.rbf_fd_solve(W, F)
        
    u_ex = u_exact(P.T)
    error = np.abs(u_soln - u_ex)
    
    # -----------------------------
    # REPORT ERRORS AND PLOT
    # -----------------------------
    X, Y = P.T
    a1 = A[0,0]
    a2 = A[1,1]
    
    # Provide error analysis from expected result
    print(f"Condition:    {np.linalg.cond(W): e}")
    print('Maximum weight:' + str(W.max()))
    print('Minimum weight:' + str(W.min()))
    print("Max error =", np.max(error))
    print("L2 error  =", np.linalg.norm(error)/np.sqrt(len(P)))
    
    # Plot a contour for the approximated solution
    plt.figure(figsize=(8, 6))
    contour_filled = plt.tricontourf(X, Y, u_soln, levels=50, cmap='viridis')
    cbar = plt.colorbar(contour_filled)
    cbar.set_label('U Solution Value', rotation=270, labelpad=15)
    plt.title(rf"""RBF-FD Approximate Solution (Contour):
    $A_{{11}}$={a1:.3e}, $A_{{22}}$={a2:.3e}""")
    plt.xlabel("x-direction")
    plt.ylabel("y-direction")
    plt.show()
    
    # Plot a 3D graph of the approximated solution
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X,Y,u_soln,cmap='viridis', edgecolor='none', alpha=0.9)
    plt.title(rf"""RBF-FD Approximate Solution:
    $A_{{11}}$={a1:.3e}, $A_{{22}}$={a2:.3e}""")
    plt.xlabel("x-direction")
    plt.ylabel("y-direction")
    plt.show()
    
    # Plot a 3D graph of the exact solution
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X,Y,u_ex,cmap='viridis', edgecolor='none', alpha=0.9)
    plt.title(rf"""Exact Solution:
    $A_{{11}}$={a1:.3e}, $A_{{22}}$={a2:.3e}""")
    plt.xlabel("x-direction")
    plt.ylabel("y-direction")
    plt.show()

    # Plot a 3D graph of the error  
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X,Y,error,cmap='viridis', edgecolor='none', alpha=0.9)
    plt.title(rf"""RBF-FD Approximate Error:
    $A_{{11}}$={a1:.3e}, $A_{{22}}$={a2:.3e}""")
    plt.xlabel("x-direction")
    plt.ylabel("y-direction")
    plt.show()

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

#%% Main Setup
if __name__ == "__main__":
    # -----------------------------
    # PARAMETERS
    # -----------------------------
    
    # Define the nodes per stencil
    num_stencil_nodes = 50
    
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
    
    P, num_int = geometry.uniform_int_square(L, Nx, Ny)
    
    # -----------------------------
    # ANISOTROPY AND PDE PROPERTIES
    # -----------------------------    
    # Define the conductivity condition
    eig_1 = 1e0
    eig_2 = 5e-3
    rad24 = 12.0
    A = coeff_matrix(eig_1, eig_2, rad24)
    print("Coefficient Matrix:\n" + str(A))
    
    # Forcing term parameters
    Amp = 1e3
    modes = [1.0,2.0]
    
    # -----------------------------
    # BUILD TEST CASE AND SOLVE
    # -----------------------------
    
    f, g, btype, u_exact = examples.example_10(Amp, modes, A)
       
    # Solve the PDE exactly and with RBF-FD
    context = assembly.rbf_fd_system(f, g, btype, P,
                                     rbf_shape, shape, L, num_stencil_nodes,
                                     num_rings, augmentation=augmentation,
                                     A=A, eps=eps, tol=tol)
    
    # Display reesults
    report_and_graph(context, u_exact)