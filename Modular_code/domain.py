# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:15:59 2026

@author: Dennis Corraliza

Stores the principal information for the RBF-FD Method
"""

class PDEDomainContext():
    """
    Mutable container holding all configuration and intermediate state
    for an RBF-FD discretization of a PDE on a fixed set of nodes.

    A `PDEDomainContext` is built once from a node set `nodes`, a set
    of precomputed stencils, and a diffusion tensor `A`, then
    incrementally populated via its `set_*` methods as the RBF-FD
    pipeline progresses: the radial basis function and its derivatives
    are configured, boundary/augmentation settings are recorded, and
    finally the assembled weight matrix and right-hand side are stored
    for solving. Functions throughout the `assembly` module take a
    `PDEDomainContext` instance and read or write its attributes
    directly rather than passing each piece of state as a separate
    argument.

    Parameters
    ----------
    nodes : numpy.ndarray, shape (num_nodes, dim)
        Coordinates of every node in the discretization.
    stencils : list of array_like
        Per-node lists of neighbor indices defining each node's local
        RBF-FD stencil (typically produced by `stencils.knn_list`).
    A : numpy.ndarray, shape (num_nodes, dim, dim)
        Diffusion tensor for the PDE operator `div(A grad u)`. Unlike
        an earlier version of this class, `A` is now a required
        constructor argument rather than defaulting to the identity
        matrix, so callers must explicitly supply it (e.g. the
        identity matrix for isotropic diffusion).

    Attributes
    ----------
    nodes : numpy.ndarray, shape (num_nodes, dim)
        Node coordinates, as passed to the constructor.
    stencils : list of array_like
        Per-node stencil index lists, as passed to the constructor.
        Can be replaced via `set_stencils`.
    center_rings : int or None
        Number of auxiliary center rings used for least-squares
        stencil weight computation (see `assembly.local_weights_ls`,
        `rbf.generate_grid_2d`). `None` selects the direct-collocation
        weight routines instead of the least-squares ones. `None`
        until set via `set_centers`.
    phi : callable or None
        Radial basis kernel function, called as `phi(p)` for a
        displacement vector `p`. `None` until set via `set_phi`.
    grad_phi : callable or None
        Gradient of the radial basis kernel, called as `grad_phi(p)`.
        `None` until set via `set_grad_phi`.
    laplacian_phi : callable or None
        Diffusion operator applied to the kernel (i.e. `div(A grad
        phi)` for the configured diffusion tensor), called as
        `laplacian_phi(p)`. `None` until set via `set_laplacian_phi`.
    augmentation : bool
        Whether RBF interpolants are augmented with a polynomial basis
        (see `rbf.poly_basis`, `rbf.grad_poly`). Defaults to `False`;
        set via `set_augmentation`.
    A : numpy.ndarray, shape (dim, dim)
        Diffusion tensor, as passed to the constructor. Can be
        replaced via `set_A`.
    W : numpy.ndarray, shape (num_nodes, num_nodes), or None
        Assembled global weight (discretization) matrix, with boundary
        conditions imposed. `None` until set via `set_W`.
    F : numpy.ndarray, shape (num_nodes,), or None
        Assembled right-hand-side vector corresponding to `W`. `None`
        until set via `set_rhs`.

    Notes
    -----
    This class performs no validation: dimensions of `stencils`, `phi`,
    `A`, `W`, and `F` are not checked against `nodes` or against each
    other, and `set_*` calls are simple attribute assignments,
    including `set_A(None)`, which will overwrite a previously valid
    `A` with `None` if called that way.
    """
        
    def __init__(self, nodes, stencils, centers, A):
        """
        Initialize the context from nodes, stencils, and a diffusion
        tensor.
    
        Sets `self.nodes`, `self.stencils`, and `self.A` directly from
        the given arguments, and initializes the remaining attributes
        either to `None` (state not yet computed: `center_rings`,
        `phi`, `grad_phi`, `laplacian_phi`, `W`, `F`) or to a sensible
        default (`augmentation = False`).
    
        Parameters
        ----------
        nodes : numpy.ndarray, shape (num_nodes, dim)
            Coordinates of every node in the discretization.
        stencils : list of array_like
            Per-node stencil index lists.
        centers : list of array_like
            Per-node stencil index lists.
        A : numpy.ndarray, shape (dim, dim)
            Diffusion tensor for the PDE operator.
    
        Returns
        -------
        None
        """
        
        self.nodes = nodes
        self.stencils = stencils
        self.centers = centers
        self.phi = None
        self.grad_phi = None
        self.laplacian_phi = None
        self.augmentation = False
        self.A = A
        self.W = None
        self.F = None
        
    def set_phi(self, phi):
        """
        Set the radial basis kernel function.

        Parameters
        ----------
        phi : callable
            Function evaluating the RBF kernel at a displacement
            vector, called as `phi(p)` (e.g. `rbf.phi_cubic` or
            `rbf.phi_gauss`).

        Returns
        -------
        None
        """
        
        self.phi = phi
        
    def set_grad_phi(self, grad_phi):
        """
        Set the gradient of the radial basis kernel function.

        Parameters
        ----------
        grad_phi : callable
            Function evaluating the gradient of the RBF kernel at a
            displacement vector, called as `grad_phi(p)` (e.g.
            `rbf.grad_phi_cubic` or `rbf.grad_phi_gauss`).

        Returns
        -------
        None
        """
        
        self.grad_phi = grad_phi
        
    def set_laplacian_phi(self, laplacian_phi):
        """
        Set the diffusion-operator-applied-to-kernel function.

        Parameters
        ----------
        laplacian_phi : callable
            Function evaluating `div(A grad phi)` at a displacement
            vector for the configured diffusion tensor, called as
            `laplacian_phi(p)` (e.g.
            `rbf.anisotropic_diffusion_phi_cubic` or
            `rbf.anisotropic_diffusion_phi_gauss`, typically wrapped in
            a closure that fixes `A`).

        Returns
        -------
        None
        """
        
        self.laplacian_phi = laplacian_phi
        
    def set_augmentation(self, augmentation):
        """
        Enable or disable polynomial augmentation of the RBF interpolant.

        Parameters
        ----------
        augmentation : bool
            Whether local RBF-FD systems should be augmented with a
            polynomial basis (see `rbf.poly_basis`, `rbf.grad_poly`) to
            exactly reproduce constant and linear functions.

        Returns
        -------
        None
        """
        
        self.augmentation = augmentation
    
    def set_A(self, A):
        """
        Replace the diffusion tensor.

        Parameters
        ----------
        A : numpy.ndarray, shape (dim, dim)
            Diffusion tensor to store. Passing `None` will overwrite
            any previously stored tensor with `None`; callers should
            ensure a valid matrix is passed if `context.A` is read
            downstream by code that does not handle `None`.

        Returns
        -------
        None
        """
        
        self.A = A
        
    def set_W(self, W):
        """
        Store the assembled global weight (discretization) matrix.

        Parameters
        ----------
        W : numpy.ndarray, shape (num_nodes, num_nodes)
            Global weight matrix with boundary conditions imposed, as
            produced by `assembly.global_weights` followed by
            `assembly.boundary_to_weights` (and possibly
            `assembly.anchor_system`).

        Returns
        -------
        None
        """
        
        self.W = W
        
    def set_rhs(self, F):
        """
        Store the assembled right-hand-side vector.

        Parameters
        ----------
        F : numpy.ndarray, shape (num_nodes,)
            Right-hand-side vector corresponding to `W`, as produced by
            `assembly.right_hand_side` (and possibly
            `assembly.anchor_system`).

        Returns
        -------
        None
        """
        
        self.F = F
