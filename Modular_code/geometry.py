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
    return np.column_stack((X,Y)), (Nx-2)*(Ny-2)

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
        `Nx + 2` nodes along x (including both endpoints).
    Ny : int
        Number of Chebyshev intervals along the y-direction; produces
        `Ny + 2` nodes along y (including both endpoints).

    Returns
    -------
    numpy.ndarray, shape ((Nx+2)*(Ny+2), 2)
        Coordinates of every grid point, in row-major (`meshgrid`/
        `ravel`) order.
    num_interior : int
        Number of interior points (`Nx * Ny`), i.e. the index
        at which the boundary points begin in `points`.

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
    x = 0.5 * L * (1 - np.cos(np.pi * np.arange(Nx) / (Nx - 1)))
    y = 0.5 * L * (1 - np.cos(np.pi * np.arange(Ny) / (Ny - 1)))
    XX, YY = np.meshgrid(x, y)
    domain = np.column_stack((XX.ravel(), YY.ravel()))
    return domain, Nx*Ny

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

def quasi_circle(R, num_rings, h_ratio=1):
    """
    Generate a 2D polar grid of auxiliary collocation centers.

    Builds a set of points filling a disk of radius `R`, arranged as
    `num_rings` concentric rings (at radii `h, 2h, ..., k*h = R`, where
    `h = R/num_rings`) plus a single point at the origin. The number of
    angular samples on each ring is chosen so that the arc-length
    spacing between neighboring points on that ring is approximately
    `h`, matching the radial spacing; this gives a roughly uniform
    (not purely radial) distribution of points across the disk.

    Parameters
    ----------
    R : float
        Radius of the disk.
    num_rings : int
        Number of concentric rings to generate, excluding the center
        point (which is always included once, in addition to the
        'num_rings').
    h_ratio : int (optional)
        Ratio of number of number of boundary nodes vs standard number
        of boundary nodes for a quasi-uniform mesh

    Returns
    -------
    numpy.ndarray, shape (nodes, 2)
        Cartesian coordinates of the generated auxiliary nodes,
        relative to the origin.
    """
    
    nodes = []
    h = R/num_rings
    rings = int(R/h)

    for i in range(1,rings):
        radius = i*h
        angle_steps = max(1, int(round(2*np.pi * radius / h)))
        
        for j in range(angle_steps):
            theta = 2*np.pi*j/angle_steps
            
            nodes.append(rad_to_euc_2d(radius, theta))

    radius = rings*h
    angle_steps = max(1, int(round(2*np.pi * radius * h_ratio / h)))

    for j in range(angle_steps):
        theta = 2*np.pi*j/angle_steps
        
        nodes.append(rad_to_euc_2d(radius, theta))

    nodes.append([0.0,0.0])
    return np.array(nodes)