# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:56:51 2026

@author: Dennis Corraliza

Defines the geomtry of the nodes in the domain and boundary
"""

import numpy as np

def uniform_square(L, Nx, Ny):
    """
    Generate a uniform tensor-product grid over a square domain.

    Builds a regular Cartesian grid of `Nx` by `Ny` points over
    `[0, L] x [0, L]` (including both boundary and interior points, with
    no separation between them) and flattens it into a single array of
    2D coordinates.

    Parameters
    ----------
    L : float
        Side length of the square domain `[0, L] x [0, L]`.
    Nx : int
        Number of grid points along the x-direction (including both
        endpoints `0` and `L`).
    Ny : int
        Number of grid points along the y-direction (including both
        endpoints `0` and `L`).

    Returns
    -------
    numpy.ndarray, shape (Nx*Ny, 2)
        Coordinates of every grid point, in row-major (`meshgrid`/
        `ravel`) order: points vary fastest along x for each fixed y.

    Notes
    -----
    Unlike `uniform_int_square`, this function does not distinguish
    between interior and boundary nodes; all `Nx*Ny` grid points
    (including the four edges) are returned together in a single
    array, and no count of interior points is returned.
    """
    
    # Nodes
    x = np.linspace(0,L,Nx)
    y = np.linspace(0,L,Ny)

    # Generate grid
    XX, YY = np.meshgrid(x,y)
    X = np.ravel(XX)
    Y = np.ravel(YY)
    return np.column_stack((X,Y))

def uniform_int_square(L, Nx_int, Ny_int, h_ratio=1):
    """
    Generate a uniform grid of interior points plus a separate set of
    boundary points on the four edges of a square domain.

    Interior points form a regular `Nx_int` by `Ny_int` grid strictly
    inside `[0, L] x [0, L]` (the outer ring of an `(Nx_int+2) x
    (Ny_int+2)` grid is dropped via slicing, so no interior point
    coincides with an edge). Boundary points are generated
    independently as `Nb` equally spaced points along each of the four
    sides (bottom, right, top, left), with duplicate corner points
    removed. The two point sets are concatenated, interior points
    first, and the number of interior points is returned alongside the
    full array so callers can split them back apart.

    Parameters
    ----------
    L : float
        Side length of the square domain `[0, L] x [0, L]`.
    Nx_int : int
        Number of interior grid points along the x-direction.
    Ny_int : int
        Number of interior grid points along the y-direction.
    h_ratio : float, optional
        Ratio between interior spacing and boundary spacing
        (default '1'); the four corners are shared between adjacent
        sides and de-duplicated, so the total number of boundary
        points returned is `h_ratio*(4*Nb - 4)` (not `h_ratio*4*Nb`).

    Returns
    -------
    points : numpy.ndarray, shape (Nx_int*Ny_int + 4*Nb - 4, 2)
        Concatenated array of interior points followed by boundary
        points.
    num_interior : int
        Number of interior points (`Nx_int * Ny_int`), i.e. the index
        at which the boundary points begin in `points`.

    Notes
    -----
    The boundary block is de-duplicated via `numpy.unique`, which
    sorts the resulting rows lexicographically by (x, y). As a result,
    the boundary points in the returned array are **not** ordered
    bottom -> right -> top -> left as they were constructed; that
    original side-by-side ordering is lost. This does not affect
    correctness elsewhere in the pipeline (boundary classification is
    done by coordinate, e.g. via `boundary.in_square_boundary`, not by
    array position), but callers should not assume any particular
    ordering of the boundary points in the output.
    """
    # Interior points (exclude boundary)
    x_int = np.linspace(0, L, Nx_int+2)[1:-1]
    y_int = np.linspace(0, L, Ny_int+2)[1:-1]
    XX, YY = np.meshgrid(x_int, y_int)
    interior = np.column_stack((XX.ravel(), YY.ravel()))

    # Boundary points
    sx = np.linspace(0, L, h_ratio*(Nx_int+2))
    sy = np.linspace(0, L, h_ratio*(Ny_int+2))

    bottom = np.column_stack((sx, np.zeros_like(sx)))
    right  = np.column_stack((L*np.ones_like(sy), sy))
    top    = np.column_stack((sx, L*np.ones_like(sx)))
    left   = np.column_stack((np.zeros_like(sy), sy))
    boundary = np.vstack((bottom, right, top, left))
    # Remove duplicate corners
    boundary = np.unique(boundary, axis=0)
    return np.vstack((interior, boundary)), len(interior)

def cheby_square(L, Nx, Ny):
    """
    Generate a tensor-product Chebyshev-Gauss-Lobatto grid over a
    square domain.

    Builds a grid whose 1D node positions along each axis follow the
    Chebyshev-Gauss-Lobatto distribution `cos(j*pi/N)` for `j = 0,
    ..., N`, affinely mapped from `[-1, 1]` to `[0, L]`. This clusters
    points near the edges (`x=0`, `x=L`, `y=0`, `y=L`) and spaces them
    more sparsely near the center, which is the standard node
    distribution for spectral/Chebyshev collocation methods. The
    resulting 1D node sets are combined into a full 2D tensor-product
    grid via `meshgrid`.

    Parameters
    ----------
    L : float
        Side length of the square domain `[0, L] x [0, L]`.
    Nx : int
        Number of Chebyshev intervals along the x-direction; produces
        `Nx + 1` nodes along x (including both endpoints).
    Ny : int
        Number of Chebyshev intervals along the y-direction; produces
        `Ny + 1` nodes along y (including both endpoints).

    Returns
    -------
    numpy.ndarray, shape ((Nx+1)*(Ny+1), 2)
        Coordinates of every grid point, in row-major (`meshgrid`/
        `ravel`) order.

    Notes
    -----
    Unlike `uniform_int_square`, this function does not separate
    interior points from boundary points (both the endpoints `0` and
    `L` are included in the 1D node sets along each axis, and no
    interior/boundary split or count is returned); the function name
    and docstring header mention "interior grid resolution" only
    because this function was adapted from `uniform_int_square` and
    `uniform_square`, but no such interior-only filtering is actually
    performed here.
    """
    # Interior points (exclude boundary)
    x_int = (np.cos(np.linspace(0, Nx, Nx+1)*np.pi/Nx)+1)*L/2
    y_int = (np.cos(np.linspace(0, Ny, Ny+1)*np.pi/Ny)+1)*L/2
    XX, YY = np.meshgrid(x_int, y_int)
    domain = np.column_stack((XX.ravel(), YY.ravel()))
    return domain


def circular_geometry(R, Nx, Ny, num_bound=100):
    """
    Generate a grid-based interior point set plus a separate ring of
    boundary points for a disk domain.

    Interior points are taken from a uniform `Nx` by `Ny` Cartesian
    grid over the bounding box `[-R, R] x [-R, R]`, keeping only those
    grid points strictly inside the disk of radius `R` (using a strict
    `<` comparison, so grid points lying exactly on the boundary
    circle are excluded from the interior set). Boundary points are
    generated independently and exactly on the circle, as `num_bound`
    equally spaced angular samples (not including the duplicate
    endpoint at `theta = 2*pi`). The interior and boundary point sets
    are concatenated, interior points first.

    Parameters
    ----------
    R : float
        Radius of the disk domain, centered at the origin.
    Nx : int
        Number of grid points along the x-direction of the bounding
        box `[-R, R]`, before filtering to the interior of the disk.
    Ny : int
        Number of grid points along the y-direction of the bounding
        box `[-R, R]`, before filtering to the interior of the disk.
    num_bound : int, optional
        Number of equally spaced points placed on the boundary circle
        (default `100`).

    Returns
    -------
    numpy.ndarray, shape (num_interior + num_bound, 2)
        Concatenated array of interior grid points (those strictly
        inside the disk) followed by boundary circle points. The
        number of interior points is not returned separately by this
        function (unlike `uniform_int_square`), so callers needing the
        split must recompute or infer it (e.g. as `len(result) -
        num_bound`).

    Notes
    -----
    Because the interior point set comes from a Cartesian grid rather
    than a disk-conforming mesh, the spacing of interior points near
    the boundary is generally not consistent with the spacing of the
    `num_bound` points placed exactly on the circle; the two point
    sets are independently generated and simply concatenated.
    """
    x = np.linspace(-R,R,Nx)
    y = np.linspace(-R,R,Ny)
    # Generating Grid
    XX, YY = np.meshgrid(x,y)
    X_ravel = np.ravel(XX)
    Y_ravel = np.ravel(YY)
    # Interior
    index_mask = (X_ravel**2+Y_ravel**2) < R**2
    X_interior = X_ravel[index_mask]
    Y_interior = Y_ravel[index_mask]
    # Boundary
    theta = np.linspace(0,2*np.pi,num_bound,endpoint=False)
    X_bound = R * np.cos(theta)
    Y_bound = R * np.sin(theta)
    # Combining Interior and Boundary
    X = list(X_interior) + list(X_bound)
    Y = list(Y_interior) + list(Y_bound)
    return np.column_stack((X,Y))