# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:39:15 2026

@author: Dennis Corraliza

Radial Basis functions and their respective elliptic operated functions
"""

import numpy as np

# theta is a list of angles for (n+1)-dimensions
def rad_to_euc(r, theta):
    """
    Convert hyperspherical coordinates to Euclidean coordinates.

    Implements the standard recursive formula for converting an
    n-sphere's (radius, angles) representation into Cartesian
    coordinates in (n+1)-dimensional space:

        x_0 = r * cos(theta_0)
        x_1 = r * sin(theta_0) * cos(theta_1)
        x_2 = r * sin(theta_0) * sin(theta_1) * cos(theta_2)
        ...
        x_{n-1} = r * sin(theta_0) * ... * sin(theta_{n-2}) * cos(theta_{n-1})
        x_n     = r * sin(theta_0) * ... * sin(theta_{n-2}) * sin(theta_{n-1})

    At each step `i`, the "remaining radius" `rho` (the magnitude of
    the as-yet-undistributed coordinate) is split into a cosine
    component (which finalizes coordinate `i`) and a sine component
    (which becomes the remaining radius carried into the next angle).
    After processing all angles in `theta`, the final remaining radius
    is appended as the last coordinate.

    For a single angle (`len(theta) == 1`), this reduces to standard
    2D polar-to-Cartesian conversion: `[r*cos(theta), r*sin(theta)]`.

    Parameters
    ----------
    r : float
        Radius (distance from the origin).
    theta : list of float
        Angles parameterizing the direction on the (n)-sphere, for a
        point in (n+1)-dimensional space. `len(theta)` determines the
        number of angles consumed; the output has `len(theta) + 1`
        coordinates.

    Returns
    -------
    list of float
        Cartesian coordinates corresponding to `(r, theta)`, of length
        `len(theta) + 1`.
    """
    
    point = [r]
    
    for i in range(len(theta)):
        rho = point[i]
        point[i] *= np.cos(theta[i])
        point.append(rho * np.sin(theta[i]))
        
    return point

def rad_to_euc_2d(r, theta):
    """
    Convert 2D radial coordinates to Euclidean coordinates.

    Implements the (radius, angle) representation into Cartesian
    coordinates in 2-dimensional space:

        x_0 = r * cos(theta)
        x_1 = r * sin(theta)

    Parameters
    ----------
    r : float
        Radius (distance from the origin).
    theta : float
        Angle parameterizing the direction on the disk.

    Returns
    -------
    list of float
        Cartesian coordinates corresponding to `(r, theta)`, of length
        `2`.
    """
            
    return [r * np.cos(theta), r * np.sin(theta)]

# Generates off-radial grid structure for robustness
# r := radius of ball
# k := number of rings (excluding center)
def generate_grid_2d(r, k):
    """
    Generate a 2D polar grid of auxiliary collocation centers.

    Builds a set of points filling a disk of radius `r`, arranged as
    `k` concentric rings (at radii `h, 2h, ..., k*h = r`, where
    `h = r/k`) plus a single point at the origin. The number of
    angular samples on each ring is chosen so that the arc-length
    spacing between neighboring points on that ring is approximately
    `h`, matching the radial spacing; this gives a roughly uniform
    (not purely radial) distribution of points across the disk,
    intended to make least-squares RBF-FD weight computations more
    robust than a purely radial sampling pattern.

    Parameters
    ----------
    r : float
        Radius of the disk (typically set to the stencil radius for a
        given node).
    k : int
        Number of concentric rings to generate, excluding the center
        point (which is always included once, in addition to the `k`
        rings).

    Returns
    -------
    numpy.ndarray, shape (num_centers, 2)
        Cartesian coordinates of the generated auxiliary centers,
        relative to the origin (i.e. these are offsets, intended to be
        added to a stencil's center node coordinate by the caller).
    """
    
    centers = []
    h = r/k
    rings = int(r/h)

    for i in range(1,rings+1):
        radius = i*h
        angle_steps = max(1, int(round(2*np.pi * radius / h)))
        
        for j in range(angle_steps):
            theta = 2*np.pi*j/angle_steps
            
            centers.append(rad_to_euc_2d(radius, theta))

    centers.append(rad_to_euc_2d(0.0,0.0))
    return np.array(centers)

def poly_basis(p):
    """
    Evaluate the linear polynomial augmentation basis at a 2D point.

    Returns the basis {1, x, y} used to augment RBF interpolants so
    that they exactly reproduce constant and linear functions
    (standard polynomial augmentation for RBF-FD methods).

    Parameters
    ----------
    p : array_like, shape (2,)
        Point (x, y) at which to evaluate the basis.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Basis values `[1.0, x, y]`.
    """
    
    x, y = p
    return np.array([1.0, x, y])

def grad_poly(p):
    """
    Evaluate the gradient of the linear polynomial basis at a 2D point.

    Returns the (constant) gradient of each basis function in
    `poly_basis` with respect to (x, y): the gradient of `1` is
    `(0, 0)`, the gradient of `x` is `(1, 0)`, and the gradient of `y`
    is `(0, 1)`.

    Parameters
    ----------
    p : array_like, shape (2,)
        Point (x, y) at which to evaluate the gradient. The value is
        unused since the gradients of {1, x, y} are constant, but the
        parameter is kept for interface consistency with other
        location-dependent callables.

    Returns
    -------
    numpy.ndarray, shape (3, 2)
        Row `i` is the gradient (d/dx, d/dy) of the `i`-th basis
        function in `poly_basis`.
    """
    
    x, y = p
    return np.array([[0.0, 0.0],
                     [1.0, 0.0],
                     [0.0, 1.0]])

def anisotropic_diffusion_poly(p,A,tol=1e-12):
    """
    Evaluate the anisotropic diffusion operator applied to the
    polynomial augmentation basis.

    Since `div(A grad phi) = 0` for every basis function in
    `poly_basis` (constants and linear terms have zero second
    derivatives, so `div(A grad phi)` vanishes regardless of `A`), this
    function always returns 0, matching the corresponding entries used
    when assembling the augmented RBF-FD system.

    Parameters
    ----------
    p : array_like, shape (2,)
        Point at which the operator would be evaluated. Unused, since
        the result is identically zero for this basis.
    A : numpy.ndarray
        Diffusion tensor.
    tol : float, optional
        Tolerance parameter, kept for interface consistency with other
        `anisotropic_diffusion_phi_*` functions. Unused (default
        `1e-12`).

    Returns
    -------
    numpy.ndarray, shape (3,)
        Always `[0.0,0.0,0.0]` for polynomial basis '[1,x,y]'.
    """
    
    return np.zeros(poly_basis(p).shape)

# Define the radial basis functions
def phi_cubic(p):  
    """
    Evaluate the cubic radial basis function at a point.

    Computes `phi(p) = ||p||^3`, the cubic RBF kernel evaluated at the
    displacement vector `p` (typically `p = x_i - x_j` between two
    nodes).

    Parameters
    ----------
    p : array_like
        Displacement vector between two points.

    Returns
    -------
    float
        The value `r**3`, where `r = ||p||` is the Euclidean norm of
        `p`.
    """
    
    r = np.sqrt(np.dot(p, p))
    
    return r**3

def grad_phi_cubic(p):
    """
    Evaluate the gradient of the cubic RBF kernel at a point.

    Computes the gradient of `phi(p) = ||p||^3` with respect to `p`,
    which simplifies to `3 * ||p|| * p`.

    Parameters
    ----------
    p : array_like
        Displacement vector between two points.

    Returns
    -------
    numpy.ndarray
        Gradient vector `3 * r * p`, where `r = ||p||`.
    """
    
    r = np.sqrt(np.dot(p, p))
    
    return 3*r*p

# Constant A
def anisotropic_diffusion_phi_cubic(p,A,tol=1e-12):
    """
    Evaluate div(A grad phi) for the cubic RBF kernel, for constant A.

    Computes the action of the (possibly anisotropic) diffusion
    operator `div(A grad phi)` on the cubic kernel `phi(p) = ||p||^3`,
    for a constant diffusion tensor `A`, in closed form:

        div(A grad phi)(p) = 3*r * ( trace(A) + (p^T A p) / r^2 )

    where `r = ||p||`. This expression has been verified symbolically
    to match `div(A grad phi)` computed directly from `phi_cubic`.
    Near the origin (`r < tol`), the function returns `0.0` to avoid
    division by zero; this is the correct limiting value, since the
    `1/r^2` singularity in the formula is exactly canceled by the
    underlying `r * (...)` behavior of the true derivative as `r -> 0`.

    Parameters
    ----------
    p : array_like, shape (dim,)
        Displacement vector between two points.
    A : numpy.ndarray, shape (dim, dim),
        Constant (symmetric) diffusion tensor.
    tol : float, optional
        Distance threshold below which `p` is treated as being at the
        origin, to avoid dividing by `r**2` (default `1e-12`).

    Returns
    -------
    float
        The value of `div(A grad phi)` at `p`, or `0.0` if
        `||p|| < tol`.
    """
    
    if A is None:
        A = np.identity(len(p))
    
    r = np.sqrt(np.dot(p, p))
    
    if r < tol:
        return 0.0
    
    return 3*r*(np.trace(A) + np.dot(p, A @ p) / r**2)

def phi_gauss(p,eps=0.5):
    """
    Evaluate the Gaussian radial basis function at a point.

    Computes `phi(p) = exp(-(eps * ||p||)^2)`, the Gaussian RBF kernel
    evaluated at the displacement vector `p`, with shape parameter
    `eps` controlling the kernel's width (larger `eps` gives a more
    sharply peaked, localized kernel).

    Parameters
    ----------
    p : array_like
        Displacement vector between two points.
    eps : float, optional
        Shape parameter of the Gaussian kernel (default `0.5`).

    Returns
    -------
    float
        The value `exp(-(eps * r)**2)`, where `r = ||p||`.
    """
    
    r2 = np.dot(p, p)   
    
    return np.exp(-(eps**2*r2))

def grad_phi_gauss(p, eps= 0.5):  
    """
    Evaluate the gradient of the Gaussian RBF kernel at a point.

    Computes the gradient of `phi(p) = exp(-(eps*||p||)^2)` with
    respect to `p`, which simplifies to `-2*eps^2 * phi(p) * p`.

    Parameters
    ----------
    p : array_like
        Displacement vector between two points.
    eps : float, optional
        Shape parameter of the Gaussian kernel, matching the value
        used in `phi_gauss` (default `0.5`).

    Returns
    -------
    numpy.ndarray
        Gradient vector `-2 * eps**2 * phi_gauss(p, eps) * p`.
    """
    
    r2 = np.dot(p, p)  
    
    return - 2 * eps **2 * np.exp(-(eps**2*r2)) * p

# Constant matrix
def anisotropic_diffusion_phi_gauss(p,A,eps=0.5):
    """
    Evaluate div(A grad phi) for the Gaussian RBF kernel, for constant A.

    Computes the action of the (possibly anisotropic) diffusion
    operator `div(A grad phi)` on the Gaussian kernel
    `phi(p) = exp(-(eps*||p||)^2)`, for a constant diffusion tensor
    `A`, in closed form:

        div(A grad phi)(p) = 2*eps^2 * phi(p) * ( 2*eps^2 * p^T A p - trace(A) )

    This expression has been verified symbolically to match
    `div(A grad phi)` computed directly from `phi_gauss`. Unlike the
    cubic kernel, no special handling is needed near `p = 0` since the
    Gaussian kernel and its derivatives are smooth everywhere.

    Parameters
    ----------
    p : array_like, shape (dim,)
        Displacement vector between two points.
    A : numpy.ndarray, shape (dim, dim),
        Constant (symmetric) diffusion tensor.
    eps : float, optional
        Shape parameter of the Gaussian kernel, matching the value
        used in `phi_gauss` (default `0.5`).

    Returns
    -------
    float
        The value of `div(A grad phi)` at `p`.
    """
     
    r2 = np.dot(p, p)  
    result = 2 * eps**2 * np.dot(p, A @ p) - np.trace(A)    
    
    return 2*eps**2*np.exp(-(eps**2*r2))*result