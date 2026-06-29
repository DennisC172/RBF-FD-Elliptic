# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:50:47 2026

@author: Dennis Corraliza

Checks and provides the boundary conditions
"""

import numpy as np

def square_corners(p, L, tol=1e-6):
    x,y = p
    
    if ((abs(x) < tol and abs(y) < tol) or
        (abs(x) < tol and abs(y-L) < tol) or
        (abs(x-L) < tol and abs(y) < tol) or
        (abs(x-L) < tol and abs(y-L) < tol)):
        return True
    
    return False

# Check whether node falls within square boundary
def in_square_boundary(p, L, btype, tol=1e-6):
    """
    Classify a node's location relative to the four sides of a square domain.

    Tests whether point `p` lies (within `tol`) on the bottom, right,
    top, or left edge of the axis-aligned square `[0, L] x [0, L]`, and
    returns the corresponding boundary-condition label from `btype`.
    The edges are checked in order: bottom (y=0), right (x=L), top
    (y=L), left (x=0); the first matching edge wins. Points matching
    none of the four edge tests are classified as `'interior'`.

    Parameters
    ----------
    p : array_like, shape (2,)
        Node coordinates (x, y) to classify.
    L : float
        Side length of the square domain `[0, L] x [0, L]`.
    btype : sequence of str, length 4
        Boundary condition type for each side, in the order
        `[bottom (y=0), right (x=L), top (y=L), left (x=0)]`.
    tol : float, optional
        Absolute tolerance used to decide whether a coordinate lies on
        an edge (default `1e-6`).

    Returns
    -------
    str
        One of `btype[0]`, `btype[1]`, `btype[2]`, `btype[3]` if `p`
        lies on the corresponding edge, or `'interior'` otherwise.

    Notes
    -----
    At a corner, two edge tests are simultaneously satisfied (e.g. the
    origin satisfies both the bottom and left edge tests). Because the
    checks are performed in a fixed order and the function returns on
    the first match, corners are always classified by the
    bottom/top test before the right/left test: the corner (0,0) and
    (L,0) take `btype[0]` (bottom), while (L,L) takes `btype[1]`
    (right) and (0,L) takes `btype[2]` (top). Callers relying on a
    specific corner classification should be aware of this precedence.
    """
    
    x, y = p
    
    if abs(y) < tol:
        return btype[0]
    if abs(x-L) < tol:
        return btype[1]
    if abs(y-L) < tol:
        return btype[2]
    if abs(x) < tol:
        return btype[3]
        
    return 'interior'

def square_normal(p, L, tol=1e-12):
    """
    Return the outward unit normal vector at a node on a square boundary.

    Determines which edge of the square `[0, L] x [0, L]` the point
    `p` lies on (within `tol`) and returns the corresponding
    axis-aligned outward unit normal: `(0, -1)` on the bottom edge,
    `(1, 0)` on the right edge, `(0, 1)` on the top edge, or `(-1, 0)`
    on the left edge.

    Parameters
    ----------
    p : array_like, shape (2,)
        Node coordinates (x, y), expected to lie on one of the four
        edges of the square.
    L : float
        Side length of the square domain `[0, L] x [0, L]`.
    tol : float, optional
        Absolute tolerance used to decide whether a coordinate lies on
        an edge (default `1e-12`).

    Returns
    -------
    numpy.ndarray, shape (2,)
        Outward unit normal vector at `p`.

    Notes
    -----
    As with `in_square_boundary`, at a corner the edge checks are
    evaluated in the same fixed order (bottom, right, top, left), so a
    corner node returns the normal of whichever edge is checked first
    among the edges it lies on, rather than e.g. a diagonal or
    averaged corner normal.

    If `p` does not lie on any edge (within `tol`), none of the
    conditions match and the function implicitly returns `None`; this
    function should only be called on nodes already classified as
    boundary nodes (e.g. via `in_square_boundary`).
    """
    
    x, y = p

    if abs(y) < tol:    # bottom
        return np.array([0.0, -1.0])
    if abs(x-L) < tol:  # right
        return np.array([1.0, 0.0])
    if abs(y-L) < tol:  # top
        return np.array([0.0, 1.0])
    if abs(x) < tol:  # left
        return np.array([-1.0, 0.0])

# g1(x) = u(x,0)
# g2(y) = u(L,y)
# g3(x) = u(x,L)
# g4(y) = u(0,y)
def square_boundary(p, L, g, tol=1e-6):
    """
    Evaluate the prescribed boundary data at a node on a square boundary.

    Determines which edge of the square `[0, L] x [0, L]` the point
    `p` lies on (within `tol`) and evaluates the corresponding
    boundary function from `g` at the appropriate coordinate:

        g1(x) = u(x, 0)   (bottom edge)
        g2(y) = u(L, y)   (right edge)
        g3(x) = u(x, L)   (top edge)
        g4(y) = u(0, y)   (left edge)

    Parameters
    ----------
    p : array_like, shape (2,)
        Node coordinates (x, y), expected to lie on one of the four
        edges of the square.
    L : float
        Side length of the square domain `[0, L] x [0, L]`.
    g : sequence of callables, length 4
        Boundary value/flux functions `[g1, g2, g3, g4]`, organized
        counterclockwise starting from the bottom edge (y=0), as
        described above.
    tol : float, optional
        Absolute tolerance used to decide whether a coordinate lies on
        an edge (default `1e-6`).

    Returns
    -------
    float
        The value of the matching boundary function evaluated at `p`'s
        relevant coordinate (`x` for the bottom/top edges, `y` for the
        right/left edges).

    Notes
    -----
    As with `in_square_boundary`, corner nodes match the edge checked
    first in the fixed evaluation order (bottom, right, top, left), so
    only one of the (generally two) applicable boundary functions is
    evaluated at a corner. If `p` does not lie on any edge (within
    `tol`), none of the conditions match and the function implicitly
    returns `None`; this function should only be called on nodes
    already classified as boundary nodes.
    """
    
    g1, g2, g3, g4 = g
    x, y = p
    
    if (abs(y) < tol):
        return g1(x)
    if (abs(x-L) < tol):
        return g2(y)
    if(abs(y-L)<tol):
        return g3(x)
    if (abs(x) < tol):
        return g4(y)

# Checks whether node falls in in circular boundary
def in_circular_boundary(p, R, btype, tol=1e-6):
    """
    Classify a node's location relative to a circular domain's boundary.

    Tests whether point `p` lies (within `tol`, measured on `x^2 + y^2
    - R^2`) on the boundary circle of radius `R` centered at the
    origin, and returns the corresponding boundary-condition label
    `btype[0]` if so, or `'interior'` otherwise.

    Parameters
    ----------
    p : array_like, shape (2,)
        Node coordinates (x, y) to classify.
    R : float
        Radius of the disk domain, centered at the origin.
    btype : sequence of str, length >= 1
        Boundary condition type to assign to boundary nodes; only
        `btype[0]` is used, since a disk has a single boundary
        component.
    tol : float, optional
        Absolute tolerance applied to `|x^2 + y^2 - R^2|` to decide
        whether a point lies on the boundary circle (default `1e-6`).

    Returns
    -------
    str
        `btype[0]` if `p` lies on the boundary circle (within `tol`),
        or `'interior'` otherwise.

    Notes
    -----
    The tolerance check is applied to `x^2 + y^2 - R^2` rather than
    directly to the radial distance `sqrt(x^2+y^2) - R`. Since
    `x^2 + y^2 - R^2 ~= 2*R*(r - R)` for `r` close to `R`, the effective
    tolerance on the actual radial distance from the boundary is
    approximately `tol / (2*R)`, which shrinks as `R` grows. Callers
    using large `R` may need to scale `tol` accordingly to keep the
    same effective geometric tolerance.
    """
    
    x, y = p
    if abs(x**2 + y**2 - R**2) < tol:
        return btype[0]
    
    return 'interior'

def disk_normal(p, L, tol=1e-12):
    """
    Return the outward unit normal vector at a node on a disk boundary.

    Computes the outward unit normal at a boundary point of a disk
    centered at the origin, which (for a circle centered at the
    origin) is simply the radial direction: `p / ||p||`.

    Parameters
    ----------
    p : array_like, shape (2,)
        Node coordinates (x, y), expected to lie on the boundary
        circle.
    L : float
        Accepted for interface consistency with `square_normal`
        (e.g. to represent the disk radius), but unused: the normal
        direction `p / ||p||` does not depend on the radius.
    tol : float, optional
        Accepted for interface consistency with `square_normal`, but
        unused, since no boundary classification or tolerance check is
        performed here (default `1e-12`).

    Returns
    -------
    numpy.ndarray, shape (2,)
        Outward unit normal vector at `p`, equal to `p / ||p||`.

    Notes
    -----
    This function does not check that `p` actually lies on the
    boundary, nor does it guard against `p` being at (or very near)
    the origin, which would cause a division by zero (or by a very
    small number). Callers should only invoke this on nodes already
    classified as boundary nodes (e.g. via `in_circular_boundary`).
    """
    
    return p / np.linalg.norm(p)

# g1(p) = u(R)
def circular_boundary(p, R, g, tol=1e-6):
    """
    Evaluate the prescribed boundary data at a node on a disk boundary.

    Checks whether `p` lies (within `tol`, measured on `x^2 + y^2 -
    R^2`) on the boundary circle of radius `R`, and if so evaluates the
    single boundary function `g` at the full point `p`.

    Parameters
    ----------
    p : array_like, shape (2,)
        Node coordinates (x, y), expected to lie on the boundary
        circle.
    R : float
        Radius of the disk domain, centered at the origin.
    g : callable
        Boundary value/flux function, called as `g(p)` (i.e. it
        receives the full 2D point, unlike `square_boundary`'s
        per-edge 1D parameterization).
    tol : float, optional
        Absolute tolerance applied to `|x^2 + y^2 - R^2|` to decide
        whether a point lies on the boundary circle (default `1e-6`).
        See the tolerance-scaling note in `in_circular_boundary`.

    Returns
    -------
    float
        The value of `g(p)` if `p` lies on the boundary circle.

    Notes
    -----
    If `p` does not lie on the boundary circle (within `tol`), the
    condition fails to match and the function implicitly returns
    `None`. This function should only be called on nodes already
    classified as boundary nodes (e.g. via `in_circular_boundary`).
    """
    
    x, y = p
    
    if (abs(x**2 + y**2 - R**2) < tol):
        return g(p)