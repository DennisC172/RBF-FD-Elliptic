# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:39:15 2026

@author: Dennis Corraliza

Radial Basis functions and their respective elliptic operated functions
"""

import numpy as np
from itertools import combinations_with_replacement

def _validate_poly_input(p, degree):
    """Validate polynomial-basis inputs and return a 2D point array."""
    p = np.asarray(p, dtype=float)
    if p.ndim not in (1, 2):
        raise ValueError("p must have shape (d,) or (n, d).")
    if p.shape[-1] == 0:
        raise ValueError("p must contain at least one spatial coordinate.")
    if not isinstance(degree, (int, np.integer)) or isinstance(degree, (bool, np.bool_)):
        raise TypeError("degree must be a nonnegative integer.")
    if degree < 0:
        raise ValueError("degree must be nonnegative.")
    single_point = p.ndim == 1
    points = p[None, :] if single_point else p
    return points, single_point


def _poly_exponents(d, degree):
    """Multi-indices for all monomials of total degree <= degree.

    Ordering preserves the original linear basis [1, x_1, ..., x_d].
    Higher degrees are ordered by total degree, then by combinations
    with replacement of coordinate indices. For d=2, degree=2 gives
    [1, x, y, x**2, x*y, y**2].
    """
    exponents = [(0,) * d]
    for total_degree in range(1, degree + 1):
        for indices in combinations_with_replacement(range(d), total_degree):
            alpha = [0] * d
            for index in indices:
                alpha[index] += 1
            exponents.append(tuple(alpha))
    return exponents


def _eval_monomial(points, alpha):
    """Evaluate x**alpha row-wise for a 2D array of points."""
    return np.prod(points ** np.asarray(alpha, dtype=int), axis=1)


def poly_basis(p, degree=1):
    """Evaluate all monomials of total degree at most ``degree``.

    The default ``degree=1`` reproduces the existing basis
    ``[1, x_1, ..., x_d]`` and therefore preserves existing callers.
    For example, in two dimensions, ``degree=2`` gives
    ``[1, x, y, x**2, x*y, y**2]`` and ``degree=3`` appends
    ``[x**3, x**2*y, x*y**2, y**3]``.

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Point or batch of points.
    degree : int, optional
        Maximum total polynomial degree (default 1).

    Returns
    -------
    numpy.ndarray
        Shape (pdim,) for one point or (n, pdim) for a batch, where
        ``pdim = comb(d + degree, degree)``.
    """
    points, single_point = _validate_poly_input(p, degree)
    exponents = _poly_exponents(points.shape[1], degree)
    values = np.column_stack([_eval_monomial(points, alpha) for alpha in exponents])
    return values[0] if single_point else values


def grad_poly(p, degree=1):
    """Evaluate gradients of the monomial basis returned by ``poly_basis``.

    The default ``degree=1`` retains the original output and ordering.

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Point or batch of points.
    degree : int, optional
        Maximum total polynomial degree (default 1).

    Returns
    -------
    numpy.ndarray
        Shape (pdim, d) for one point or (n, pdim, d) for a batch.
        Entry ``[..., k]`` is the derivative with respect to coordinate k.
    """
    points, single_point = _validate_poly_input(p, degree)
    n, d = points.shape
    exponents = _poly_exponents(d, degree)
    gradients = np.zeros((n, len(exponents), d), dtype=float)

    for basis_index, alpha in enumerate(exponents):
        for k, alpha_k in enumerate(alpha):
            if alpha_k == 0:
                continue
            beta = list(alpha)
            beta[k] -= 1
            gradients[:, basis_index, k] = alpha_k * _eval_monomial(points, beta)

    return gradients[0] if single_point else gradients


def anisotropic_diffusion_poly(p, A=None, degree=1):
    r"""Evaluate ``div(A grad(q))`` for every monomial q in the basis.

    This is the constant-coefficient operator used by the existing RBF
    routines:

        div(A grad(q)) = sum_{i,j} A[i,j] * d^2 q/(dx_i dx_j).

    For degree 0 or 1, the result is identically zero and ``A`` may be
    omitted, preserving the original ``anisotropic_diffusion_poly(p)``
    call. For degree >= 2, ``A`` must be supplied; quadratic and higher
    monomials generally have nonzero operator values.

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Point or batch of points.
    A : array_like, shape (d, d), optional
        Constant diffusion tensor. Required when ``degree >= 2``.
    degree : int, optional
        Maximum total polynomial degree, matching ``poly_basis`` and
        ``grad_poly`` (default 1).

    Returns
    -------
    numpy.ndarray
        Shape (pdim,) for one point or (n, pdim) for a batch. The
        ordering exactly matches ``poly_basis(p, degree)``.

    Notes
    -----
    This function assumes A is constant, just like
    ``anisotropic_diffusion_phi_cubic`` and the other kernel-operator
    routines in this module. For variable A(x), the additional term
    ``(div A) dot grad(q)`` must also be included.
    """
    points, single_point = _validate_poly_input(p, degree)
    n, d = points.shape
    exponents = _poly_exponents(d, degree)
    result = np.zeros((n, len(exponents)), dtype=float)

    if degree >= 2:
        if A is None:
            raise ValueError("A must be supplied when degree >= 2.")
        A = np.asarray(A, dtype=float)
        if A.shape != (d, d):
            raise ValueError(f"A must have shape ({d}, {d}); received {A.shape}.")

        # Apply the Hessian contraction sum_{i,j} A[i,j] * d_ij(q).
        # Summing over ordered pairs (i,j) correctly includes both
        # mixed-derivative contributions for symmetric A.
        for basis_index, alpha in enumerate(exponents):
            for i in range(d):
                for j in range(d):
                    if i == j:
                        if alpha[i] < 2:
                            continue
                        beta = list(alpha)
                        coefficient = A[i, i] * alpha[i] * (alpha[i] - 1)
                        beta[i] -= 2
                    else:
                        if alpha[i] < 1 or alpha[j] < 1:
                            continue
                        beta = list(alpha)
                        coefficient = A[i, j] * alpha[i] * alpha[j]
                        beta[i] -= 1
                        beta[j] -= 1
                    result[:, basis_index] += coefficient * _eval_monomial(points, beta)

    return result[0] if single_point else result


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


def phi_quintic(p):
    """Evaluate the quintic polyharmonic RBF ``phi(p) = ||p||**5``.

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Displacement vector(s).

    Returns
    -------
    float or numpy.ndarray, shape (n,)
        Scalar value for one displacement, or one value per batch row.
    """

    p = np.asarray(p, dtype=float)
    r = np.sqrt(np.sum(p**2, axis=-1))
    return r**5


def grad_phi_quintic(p):
    """Evaluate the gradient of ``phi(p) = ||p||**5`` with respect to p.

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Displacement vector(s).

    Returns
    -------
    numpy.ndarray, shape (d,) or (n, d)
        ``5 * ||p||**3 * p``, evaluated row-wise for batched input.
    """

    p = np.asarray(p, dtype=float)
    r = np.sqrt(np.sum(p**2, axis=-1))

    if p.ndim == 1:
        return 5*r**3*p
    return 5*r[:, None]**3*p


def anisotropic_diffusion_phi_quintic(p, A, tol=1e-12):
    """Evaluate ``div(A grad(phi))`` for ``phi(p) = ||p||**5``.

    A is assumed constant, as in ``anisotropic_diffusion_phi_cubic``.
    The analytic expression is

        5 * r**3 * trace(A) + 15 * r * (p.T @ A @ p),

    where ``r = ||p||``.  This form has no division by r and therefore
    is well-defined at p = 0 (where the result is zero).  ``tol`` is
    accepted for call-signature compatibility with the cubic routine;
    it is not needed by this nonsingular formula.

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Displacement vector(s).
    A : array_like, shape (d, d)
        Constant diffusion tensor.
    tol : float, optional
        Retained for interface compatibility; unused.

    Returns
    -------
    float or numpy.ndarray, shape (n,)
        Operator value for one displacement, or one value per batch row.
    """

    p = np.asarray(p, dtype=float)
    A = np.asarray(A, dtype=float)
    r = np.sqrt(np.sum(p**2, axis=-1))
    trA = np.trace(A)

    if p.ndim == 1:
        quad = np.dot(p, A @ p)
    else:
        quad = np.einsum('ij,jk,ik->i', p, A, p)

    return 5*r**3*trA + 15*r*quad


def phi_septic(p):
    """Evaluate the septic polyharmonic RBF ``phi(p) = ||p||**7``.

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Displacement vector(s).

    Returns
    -------
    float or numpy.ndarray, shape (n,)
        Scalar value for one displacement, or one value per batch row.
    """

    p = np.asarray(p, dtype=float)
    r = np.sqrt(np.sum(p**2, axis=-1))
    return r**7


def grad_phi_septic(p):
    """Evaluate the gradient of ``phi(p) = ||p||**7`` with respect to p.

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Displacement vector(s).

    Returns
    -------
    numpy.ndarray, shape (d,) or (n, d)
        ``7 * ||p||**5 * p``, evaluated row-wise for batched input.
    """

    p = np.asarray(p, dtype=float)
    r = np.sqrt(np.sum(p**2, axis=-1))

    if p.ndim == 1:
        return 7*r**5*p
    return 7*r[:, None]**5*p


def anisotropic_diffusion_phi_septic(p, A, tol=1e-12):
    """Evaluate ``div(A grad(phi))`` for ``phi(p) = ||p||**7``.

    A is assumed constant, as in ``anisotropic_diffusion_phi_cubic``.
    The analytic expression is

        7 * r**5 * trace(A) + 35 * r**3 * (p.T @ A @ p),

    where ``r = ||p||``.  This form has no division by r and therefore
    is well-defined at p = 0 (where the result is zero).  ``tol`` is
    accepted for call-signature compatibility with the cubic routine;
    it is not needed by this nonsingular formula.

    Parameters
    ----------
    p : array_like, shape (d,) or (n, d)
        Displacement vector(s).
    A : array_like, shape (d, d)
        Constant diffusion tensor.
    tol : float, optional
        Retained for interface compatibility; unused.

    Returns
    -------
    float or numpy.ndarray, shape (n,)
        Operator value for one displacement, or one value per batch row.
    """

    p = np.asarray(p, dtype=float)
    A = np.asarray(A, dtype=float)
    r = np.sqrt(np.sum(p**2, axis=-1))
    trA = np.trace(A)

    if p.ndim == 1:
        quad = np.dot(p, A @ p)
    else:
        quad = np.einsum('ij,jk,ik->i', p, A, p)

    return 7*r**5*trA + 35*r**3*quad

def stencil_eps(diff_P, coeff=0.050, tol=1e-12):
    dist_M = np.linalg.norm(diff_P, axis=-1)
    sep = dist_M[dist_M > tol].max()

    return coeff/sep

def phi_gauss(p,eps):
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
    eps : float
        Shape parameter of the Gaussian kernel.

    Returns
    -------
    float or numpy.ndarray, shape (n,)
        The value(s) `exp(-(eps * r)**2)`, where `r = ||p||`, computed
        per-row for batched input.
    """
    
    p = np.asarray(p, dtype=float)
    r2 = np.sum(p**2, axis=-1)
    
    return np.exp(-(eps**2*r2))

def grad_phi_gauss(p, eps):  
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
    eps : float
        Shape parameter of the Gaussian kernel, matching the value
        used in `phi_gauss`.

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
def anisotropic_diffusion_phi_gauss(p,A,eps):
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
    eps : float
        Shape parameter of the Gaussian kernel, matching the value
        used in `phi_gauss`.

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