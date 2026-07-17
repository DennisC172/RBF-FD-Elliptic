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
    Evaluate the linear polynomial augmentation basis at one or more
    points, in any spatial dimension.

    Returns the basis {1, x_1, ..., x_d} used to augment RBF
    interpolants so that they exactly reproduce constant and linear
    functions (standard polynomial augmentation for RBF-FD methods).

    Accepts either a single point or a batch of points:

    - `p` of shape `(d,)` (a single point): returns a `(d+1,)` vector
      `[1.0, x_1, ..., x_d]`.
    - `p` of shape `(n, d)` (a batch of `n` points): returns an
      `(n, d+1)` matrix whose row `i` is `[1.0, p[i,0], ..., p[i,d-1]]`.

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Single point or batch of points at which to evaluate the
        basis, in `d` spatial dimensions.

    Returns
    -------
    numpy.ndarray, shape (d+1,) or (n, d+1)
        Basis values `[1.0, x_1, ..., x_d]` per point, batched along
        the leading axis if `p` was 2D.

    Notes
    -----
    Unlike `phi_cubic`/`phi_gauss` and friends (which *reduce* a
    displacement vector's trailing axis to a scalar via a norm),
    `poly_basis` *expands* each point into `d+1` basis values, so
    batching stacks along a new leading axis rather than collapsing
    the trailing one. This is a different vectorization pattern from
    the kernel functions below, even though both accept the same
    `(d,)`/`(n, d)` input convention.
    """
    
    p = np.asarray(p, dtype=float)
    
    if p.ndim == 1:
        return np.concatenate([[1.0], p])
    
    n = p.shape[0]
    return np.column_stack([np.ones(n), p])

def grad_poly(p):
    """
    Evaluate the gradient of the linear polynomial basis at one or
    more points, in any spatial dimension.

    Returns the (constant) gradient of each basis function in
    `poly_basis` with respect to the spatial coordinates: the gradient
    of the constant term `1` is the zero vector, and the gradient of
    coordinate function `x_k` is the `k`-th standard basis vector.
    Since these gradients don't depend on `p`'s value, batching only
    changes the output's shape (by broadcasting the same constant
    block across the batch axis), not its content.

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Single point or batch of `n` points, in `d` spatial
        dimensions. The values are unused since the gradients of
        `{1, x_1, ..., x_d}` are constant, but the shape of `p`
        determines whether a single block or a batched stack of
        (identical) blocks is returned.

    Returns
    -------
    numpy.ndarray, shape (d+1, d) or (n, d+1, d)
        For a single point: row `i` is the gradient of the `i`-th
        basis function in `poly_basis`. For a batch of `n` points: the
        same `(d+1, d)` block is broadcast (not copied with
        per-point differences, since there are none) across a new
        leading axis of length `n`.
    """
    
    p = np.asarray(p, dtype=float)
    d = p.shape[-1]
    
    base = np.vstack([np.zeros(d), np.eye(d)])  # shape (d+1, d)
    
    if p.ndim == 1:
        return base
    
    n = p.shape[0]
    return np.broadcast_to(base, (n, d+1, d))

def anisotropic_diffusion_poly(p):
    """
    Evaluate the anisotropic diffusion operator applied to the
    polynomial augmentation basis, at one or more points.

    Since `div(A grad phi) = 0` for every basis function in
    `poly_basis` (constants and linear terms have zero second
    derivatives, so `div(A grad phi)` vanishes regardless of `A`), this
    function always returns an all-zeros vector of length `d+1` (one
    entry per basis function), matching `poly_basis`'s output
    convention and the corresponding entries used when assembling the
    augmented RBF-FD system.

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Point or batch of `n` points at which the operator would be
        evaluated. Only used to infer `d` (and, for batched input,
        `n`); the result is identically zero for this basis regardless
        of `p`'s value.
    A : numpy.ndarray
        Diffusion tensor. Unused, since the result is identically zero
        regardless of `A` for a degree-1 polynomial basis.
    tol : float, optional
        Tolerance parameter, kept for interface consistency with other
        `anisotropic_diffusion_phi_*` functions. Unused (default
        `1e-12`).

    Returns
    -------
    numpy.ndarray, shape (d+1,) or (n, d+1)
        All-zeros array, shaped to match `poly_basis(p)`'s output so
        it can be concatenated directly with the kernel-side RHS block
        in the augmented system (see `assembly.local_weights_solve`
        and friends).

    Notes
    -----
    This result depends only on the *degree* of `poly_basis` (1:
    constant + linear), not on `A`. If `poly_basis`/`grad_poly` are
    ever extended to include quadratic terms, this function would need
    to return actual nonzero values (proportional to `A`'s entries)
    rather than zeros, since `div(A grad phi)` no longer vanishes for
    quadratics. The output shape would stay `(pdim,)`/`(n, pdim)` with
    `pdim` matching the extended basis size.
    """
    
    p = np.asarray(p, dtype=float)
    d = p.shape[-1]
    
    if p.ndim == 1:
        return np.zeros(d+1)
    
    n = p.shape[0]
    return np.zeros((n, d+1))

# Define the radial basis functions
def phi_cubic(p):  
    """
    Evaluate the cubic radial basis function at one or more points.

    Computes `phi(p) = ||p||^3`, the cubic RBF kernel evaluated at the
    displacement vector(s) `p` (typically `p = x_i - x_j` between two
    nodes).

    Accepts either a single displacement vector or a batch:

    - `p` of shape `(d,)`: returns a scalar.
    - `p` of shape `(n, d)`: returns an `(n,)` array, with each row's
      norm computed independently (the trailing axis `d` is reduced;
      the leading batch axis `n` is left untouched).

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Displacement vector, or batch of `n` displacement vectors,
        between two points.

    Returns
    -------
    float or numpy.ndarray, shape (n,)
        The value(s) `r**3`, where `r = ||p||` is the Euclidean norm
        of `p`, computed per-row for batched input.
    """
    
    p = np.asarray(p, dtype=float)
    r = np.sqrt(np.sum(p**2, axis=-1))
    
    return r**3

def grad_phi_cubic(p):
    """
    Evaluate the gradient of the cubic RBF kernel at one or more points.

    Computes the gradient of `phi(p) = ||p||^3` with respect to `p`,
    which simplifies to `3 * ||p|| * p`.

    Accepts either a single displacement vector or a batch:

    - `p` of shape `(d,)`: returns a `(d,)` gradient vector.
    - `p` of shape `(n, d)`: returns an `(n, d)` array, with each row's
      gradient computed from that row's own norm only (no mixing
      between rows).

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Displacement vector, or batch of `n` displacement vectors,
        between two points.

    Returns
    -------
    numpy.ndarray, shape (d,) or (n, d)
        Gradient vector(s) `3 * r * p`, where `r = ||p||`, computed
        per-row for batched input.
    """
    
    p = np.asarray(p, dtype=float)
    r = np.sqrt(np.sum(p**2, axis=-1))
    
    if p.ndim == 1:
        return 3*r*p
    
    return 3*r[:, None]*p

# Constant A
def anisotropic_diffusion_phi_cubic(p,A,tol=1e-12):
    """
    Evaluate div(A grad phi) for the cubic RBF kernel, for constant A,
    at one or more points.

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

    Accepts either a single displacement vector or a batch:

    - `p` of shape `(d,)`: returns a scalar.
    - `p` of shape `(n, d)`: returns an `(n,)` array, with each row's
      quadratic form `p^T A p` computed independently via `einsum`
      (row `i`'s result only ever involves row `i`'s own `p`, never
      another row's, unlike a plain `p @ A @ p.T` which would produce
      an `(n, n)` matrix of cross terms).

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Displacement vector, or batch of `n` displacement vectors,
        between two points.
    A : numpy.ndarray, shape (d, d),
        Constant (symmetric) diffusion tensor.
    tol : float, optional
        Distance threshold below which a row of `p` is treated as
        being at the origin, to avoid dividing by `r**2` (default
        `1e-12`).

    Returns
    -------
    float or numpy.ndarray, shape (n,)
        The value(s) of `div(A grad phi)` at `p`, or `0.0` for any row
        with `||p|| < tol`, computed per-row for batched input.
    """
    
    p = np.asarray(p, dtype=float)
        
    r = np.sqrt(np.sum(p**2, axis=-1))
    trA = np.trace(A)
    
    if p.ndim == 1:
        if r < tol:
            return 0.0
        return 3*r*(trA + np.dot(p, A @ p) / r**2)
    
    quad = np.einsum('ij,jk,ik->i', p, A, p)
    r_safe = np.where(r < tol, 1.0, r)  # avoid 0/0 before the outer where selects it away
    result = 3*r*(trA + quad/r_safe**2)
    return np.where(r < tol, 0.0, result)

def phi_gauss(p,eps=0.5):
    """
    Evaluate the Gaussian radial basis function at one or more points.

    Computes `phi(p) = exp(-(eps * ||p||)^2)`, the Gaussian RBF kernel
    evaluated at the displacement vector(s) `p`, with shape parameter
    `eps` controlling the kernel's width (larger `eps` gives a more
    sharply peaked, localized kernel).

    Accepts either a single displacement vector or a batch:

    - `p` of shape `(d,)`: returns a scalar.
    - `p` of shape `(n, d)`: returns an `(n,)` array, with each row's
      norm computed independently (the trailing axis `d` is reduced;
      the leading batch axis `n` is left untouched, so rows are never
      normed together).

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Displacement vector, or batch of `n` displacement vectors,
        between two points.
    eps : float, optional
        Shape parameter of the Gaussian kernel (default `0.5`).

    Returns
    -------
    float or numpy.ndarray, shape (n,)
        The value(s) `exp(-(eps * r)**2)`, where `r = ||p||`, computed
        per-row for batched input.
    """
    
    p = np.asarray(p, dtype=float)
    r2 = np.sum(p**2, axis=-1)
    
    return np.exp(-(eps**2*r2))

def grad_phi_gauss(p, eps=0.5):  
    """
    Evaluate the gradient of the Gaussian RBF kernel at one or more
    points.

    Computes the gradient of `phi(p) = exp(-(eps*||p||)^2)` with
    respect to `p`, which simplifies to `-2*eps^2 * phi(p) * p`.

    Accepts either a single displacement vector or a batch:

    - `p` of shape `(d,)`: returns a `(d,)` gradient vector.
    - `p` of shape `(n, d)`: returns an `(n, d)` array, with each row's
      gradient scaled by that row's own `phi` value only (broadcast via
      `phi[:, None]`, so no row's scalar prefactor leaks into another
      row's vector).

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Displacement vector, or batch of `n` displacement vectors,
        between two points.
    eps : float, optional
        Shape parameter of the Gaussian kernel, matching the value
        used in `phi_gauss` (default `0.5`).

    Returns
    -------
    numpy.ndarray, shape (d,) or (n, d)
        Gradient vector(s) `-2 * eps**2 * phi_gauss(p, eps) * p`,
        computed per-row for batched input.
    """
    
    p = np.asarray(p, dtype=float)
    r2 = np.sum(p**2, axis=-1)
    phi = np.exp(-(eps**2*r2))
    
    if p.ndim == 1:
        return -2*eps**2*phi*p
    
    return -2*eps**2*phi[:, None]*p

# Constant matrix
def anisotropic_diffusion_phi_gauss(p,A,eps=0.5):
    """
    Evaluate div(A grad phi) for the Gaussian RBF kernel, for constant
    A, at one or more points.

    Computes the action of the (possibly anisotropic) diffusion
    operator `div(A grad phi)` on the Gaussian kernel
    `phi(p) = exp(-(eps*||p||)^2)`, for a constant diffusion tensor
    `A`, in closed form:

        div(A grad phi)(p) = 2*eps^2 * phi(p) * ( 2*eps^2 * p^T A p - trace(A) )

    This expression has been verified symbolically to match
    `div(A grad phi)` computed directly from `phi_gauss`. Unlike the
    cubic kernel, no special handling is needed near `p = 0` since the
    Gaussian kernel and its derivatives are smooth everywhere.

    Accepts either a single displacement vector or a batch:

    - `p` of shape `(d,)`: returns a scalar.
    - `p` of shape `(n, d)`: returns an `(n,)` array, with each row's
      quadratic form `p^T A p` computed independently via `einsum`
      (row `i`'s result only ever involves row `i`'s own `p`, never
      another row's).

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Displacement vector, or batch of `n` displacement vectors,
        between two points.
    A : numpy.ndarray, shape (d, d),
        Constant (symmetric) diffusion tensor.
    eps : float, optional
        Shape parameter of the Gaussian kernel, matching the value
        used in `phi_gauss` (default `0.5`).

    Returns
    -------
    float or numpy.ndarray, shape (n,)
        The value(s) of `div(A grad phi)` at `p`, computed per-row for
        batched input.
    """
     
    p = np.asarray(p, dtype=float)
    r2 = np.sum(p**2, axis=-1)
    phi = np.exp(-(eps**2*r2))
    trA = np.trace(A)
    
    if p.ndim == 1:
        quad = np.dot(p, A @ p)
    else:
        quad = np.einsum('ij,jk,ik->i', p, A, p)
    
    result = 2*eps**2*quad - trA
    
    return 2*eps**2*phi*result