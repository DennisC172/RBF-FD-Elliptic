# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:43:57 2026

@author: Dennis Corraliza

Example Case Studies for Anisotropic Elliptic PDEs
"""

import numpy as np
import scipy.integrate as integrate

# Square domain example
# Solution for an isotropic PDE with forcing term and boundary conditions
def example_1(eig_1=None,eig_2=None,angle=None,Amp=1.0,modes=[1.0,1.0],L=1.0):
    """
    Poisson problem on the unit square. 
    ISOTROPIC Problem.
    
    PDE:
        .. math::
            \\Delta u = f \quad \text{in } \Omega = [0,L]^2.
    
    Boundary conditions:
        .. math::
            u(x,0) = v(x), \quad
            u(0,y) = y - y^2 / L, \quad
            u = 0 \text{ on the remaining sides}.
    
        where the sharp-hat function on y=0 is
    
        .. math::
            v(x) =
            \\begin{cases}
                x, & x \le L/2, \\\\
                L-x, & x > L/2.
            \\end{cases}
    
    Exact solution:
        .. math::
            u(x,y) = u_p(x,y) + u_h(x,y),
    
        with the particular solution
    
        .. math::
            u_p(x,y) =
            \\frac{A}{\\alpha^2 + \\beta^2}
            \sin(\alpha\pi x / L)\sin(\beta\pi y / L),
    
        and u_h constructed from a Fourier series to enforce the
        non-trivial boundary data on x=0 and y=0.
    
    Returns
    -------
    f : callable
        Right-hand side function f(x,y).
    g : list of callables
        Boundary condition functions ordered as [y=0, x=L, y=L, x=0].
    btype : list of str
        Boundary condition types (all Dirichlet in this example).
    u_exact : callable
        Exact solution u(x,y).
    """
        
    def evaluate_sharp_hat(x):
        """Computes the sharp hat boundary condition at y=0 safely."""
        # Ensure input stays inside the analytical boundaries [0, L]            
        if x <= L/2:
            return x
        else:
            return L - x
    
    def u_exact(p, fourier_terms=100):
        """
        Evaluates the exact solution u(x,y) for the Poisson problem.
        Uses clean if/else defensive logic instead of try/except.
        """
        x,y = p
        alpha, beta = modes
    
        # 2. Particular solution segment evaluation        
        A_coef = Amp / (alpha**2 + beta**2)
        u_particular = A_coef*np.sin(alpha*np.pi*x/L)*np.sin(beta*np.pi*y/L)
        
        # 3. Homogeneous Fourier Series evaluation for the boundaries
        # Evaluates the boundary mappings g(y) = y-y^2/L and the sharp hat fnc
        u_homogeneous = 0.0
        
        for n in range(1, fourier_terms + 1):
            # Fourier coefficients for g(y) = y - y^2
            # Integral calculation: 2 * over_domain( (y-y^2/L) * sin(n*pi*y/L))
            if n % 2 == 0:
                a_n = 0 # Even terms cancel out due to symmetry
            else:
                a_n = 8 * L / ((n * np.pi) ** 3)
                
            # Fourier coefficients for the sharp hat function with height L/2
            if n % 2 == 0:
                b_n = 0
            else:
                # Alternate signs for peak convergence
                k = (n - 1) // 2
                b_n = (4 * L / ((n * np.pi) ** 2)) * ((-1) ** k)
    
            # Build stable hyperbolic scaling arrays for boundary matching
            # Mapping the x=0 boundary data across the x-axis
            u_homogeneous += a_n*(np.sinh(n*np.pi*(L-x)/L)/np.sinh(n*np.pi)*
                                  np.sin(n * np.pi * y / L))
            
            # Mapping the y=0 sharp hat boundary data across the y-axis
            u_homogeneous += b_n*(np.sinh(n*np.pi*(L-y)/L)/np.sinh(n*np.pi)*
                                  np.sin(n * np.pi * x / L))
    
        return u_particular + u_homogeneous
    
    # RHS function
    def f(p):
        x, y = p
        alpha, beta = modes
        
        return - (Amp*np.pi**2/L**2*np.sin(alpha*np.pi*x/L)*
                                    np.sin(beta*np.pi*y/L))
    
    # Boundary condition
    g = [lambda x: x if 0<=x<=L/2 else L-x,
         lambda y: 0.0,
         lambda x: 0.0,
         lambda y: y-y**2/L
         ]
    
    btype = [
        'dirichlet',
        'dirichlet',
        'dirichlet',
        'dirichlet'        
    ]
    
    return f, g, btype, u_exact

# Solutions with anisotropy, 0 boundary and mainly the forcing term
def example_2(eig_1=None,eig_2=None,angle=None,Amp=1.0,modes=[1.0,1.0],L=1.0):
    """
    Anisotropic diffusion problem on the square domain [0, L] × [0, L]
    with homogeneous Dirichlet boundary conditions.

    PDE:
        .. math::
            \\nabla \\cdot (A \\nabla u) = f
            \quad \text{in } [0,L]^2,

        where A is a symmetric positive definite diffusion tensor.

    Exact solution:
        .. math::
            u(x,y) =
            A_c \sin(\\alpha \\pi x / L)\sin(\\beta \\pi y / L)
            + B_c \sin(\\beta \\pi x / L)\sin(\\alpha \\pi y / L),

        with coefficients

        .. math::
            A_c = \\frac{\\text{Amp}}
            {\\pi^2 (\\alpha^2 A_{11} + \\beta^2 A_{22}) / L^2},
            \qquad
            B_c = \\frac{\\text{Amp}}
            {\\pi^2 (\\beta^2 A_{11} + \\alpha^2 A_{22}) / L^2}.

        These ensure the manufactured solution satisfies the anisotropic
        operator for any SPD matrix A and any domain size L.

    Boundary conditions:
        .. math::
            u = 0 \quad \text{on } \partial[0,L]^2.

    Forcing:
        Computed analytically from the anisotropic operator and includes
        diagonal and mixed-derivative contributions.

    Returns
    -------
    f : callable
        Right-hand side f(x,y).
    g : list of callables
        Boundary data (all zero).
    btype : list of str
        Boundary condition types (all Dirichlet).
    u_exact : callable
        Exact solution u(x,y).
    """

    def u_exact(p,A):
        x, y = p
        alpha, beta = modes

        # Scaling factor for domain size L
        kx = alpha * np.pi / L
        ky = beta  * np.pi / L

        # Correct coefficients for general L
        A_coef = Amp / (kx**2 * A[0,0] + ky**2 * A[1,1])
        B_coef = Amp / (ky**2 * A[0,0] + kx**2 * A[1,1])

        return (
            A_coef * np.sin(kx * x) * np.sin(ky * y)
            + B_coef * np.sin(ky * x) * np.sin(kx * y)
        )

    def f(p,A):
        x, y = p
        alpha, beta = modes

        kx = alpha * np.pi / L
        ky = beta  * np.pi / L

        # Mixed derivative coefficients
        C_coeff = -2 * kx * ky * A[0,1] / (kx**2 * A[0,0] + ky**2 * A[1,1])
        D_coeff = -2 * kx * ky * A[0,1] / (ky**2 * A[0,0] + kx**2 * A[1,1])

        return -Amp * (
            np.sin(kx * x) * np.sin(ky * y)
            + np.sin(ky * x) * np.sin(kx * y)
            + C_coeff * np.cos(kx * x) * np.cos(ky * y)
            + D_coeff * np.cos(ky * x) * np.cos(kx * y)
        )

    # Homogeneous Dirichlet on all boundaries
    g = 4 * [lambda x: 0.0]

    btype = ['dirichlet'] * 4

    return f, g, btype, u_exact

def example_3(eig_1=None,eig_2=None,angle=None,Amp=1.0,modes=[1.0,1.0],L=1.0):
    """
    Anisotropic diffusion problem with mixed polynomial–trigonometric solution.

    PDE:
        .. math::
            \\nabla\\cdot(A\\nabla u) = f
            \quad \text{in } [0,L]^2.

    Exact solution:
        .. math::
            u(x,y) =
            (L-x)(y-y^2/L)
            + c(L-y)x(L-x)/L
            + \\text{Amp}\,\sin(\\alpha\pi x/L)\sin(\\beta\pi y/L),

        where ``c = 2`` and ``modes = [alpha, beta]``.

    Boundary conditions (Dirichlet):
        .. math::
            u(x,0) = c\,x(L-x), \quad
            u(L,y) = 0, \quad
            u(x,L) = 0, \quad
            u(0,y) = y - y^2/L.

    Forcing:
        Includes polynomial, trig, and anisotropic cross-derivative terms.

    Returns
    -------
    f : callable
    g : list of callables
    btype : list of str
    u_exact : callable
    """

    coeff_y_0 = 2

    def u_exact(p,A=None):
        x, y = p
        alpha, beta = modes

        # Polynomial part scaled for [0,L]^2
        poly = (
            (L - x) * (y - y**2 / L)
            + coeff_y_0 * (L - y) * x * (L - x) / L
        )

        # Trigonometric part with correct scaling
        kx = alpha * np.pi / L
        ky = beta  * np.pi / L

        trig = Amp * np.sin(kx * x) * np.sin(ky * y)

        return poly + trig

    def f(p,A):
        x, y = p
        alpha, beta = modes

        A11 = A[0,0]
        A12 = A[0,1]
        A22 = A[1,1]

        kx = alpha * np.pi / L
        ky = beta  * np.pi / L

        # Polynomial Laplacian terms (scaled for L)
        poly_term = (
            -coeff_y_0*2*A11*(L-y)/L
            +2*A12*(coeff_y_0*2*x/L+2*y/L-(coeff_y_0+1)/L)
            - 2*A22*(L-x)/L
        )

        # Trigonometric Laplacian
        trig_term = (
            -Amp*(A11*kx**2+A22*ky**2)
            *np.sin(kx*x)*np.sin(ky*y)
        )

        # Mixed derivative term
        mixed_term = (
            2*Amp*A12*kx*ky
            *np.cos(kx*x)*np.cos(ky*y)
        )

        return poly_term + trig_term + mixed_term

    # Dirichlet boundary conditions for general L
    g = [
        lambda x: coeff_y_0 * x * (L - x),     # y = 0
        lambda y: 0.0,                         # x = L
        lambda x: 0.0,                         # y = L
        lambda y: y - y**2 / L                 # x = 0
    ]

    btype = ['dirichlet'] * 4

    return f, g, btype, u_exact


def example_4(eig_1=None,eig_2=None,angle=None,Amp=1.0,modes=None,L=1.0):
    """
    Constant manufactured solution for the Poisson equation on [0,L]×[0,L].

    PDE:
        .. math::
            \\nabla\\cdot(A\\nabla u) = 0.0.

    Exact solution:
        .. math::
            u(x,y) = 1.

    Boundary conditions:
        Dirichlet: u = 1 on all sides.

    Returns
    -------
    f, g, btype, u_exact
    """
    
    def u_exact(p,A):
        return np.ones(p.shape[-1])
    
    def f(p,A):
       return 0.0
    
    g = [
        lambda x: 1.0,   # y=0
        lambda y: 1.0,   # x=L
        lambda x: 1.0,   # y=L
        lambda y: 1.0    # x=0
    ]
    
    btype = [
        'dirichlet',
        'dirichlet',
        'dirichlet',
        'dirichlet'        
    ]
    
    return f, g, btype, u_exact

def example_5(eig_1=None,eig_2=None,angle=None,Amp=1.0,modes=None,L=1.0):
    """
    Quadratic manufactured solution for anisotropic diffusion on [0,L]×[0,L].

    Exact solution:
        .. math::
            u(x,y) = x^2 + y^2.

    PDE:
        .. math::
            -\\nabla\\cdot(A\\nabla u) = 2\,\\mathrm{trace}(A).

    Boundary conditions:
        Dirichlet values matching the quadratic polynomial.

    Returns
    -------
    f : callable
        Right-hand side :math:`f(x,y)`.
    g : list of callables
        Dirichlet boundary data in the order [y=0, x=L, y=L, x=0].
    btype : list of str
        Boundary condition types (all Dirichlet).
    u_exact : callable
        Exact solution :math:`u(x,y)`.
    """
    
    def u_exact(p,A):        
        return np.linalg.norm(p,axis=0)**2
    
    def f(p,A):
        return 2*np.linalg.trace(A)
    
    g = [
        lambda x: x**2,         # y = 0
        lambda y: L**2 + y**2,  # x = L
        lambda x: L**2 + x**2,  # y = L
        lambda y: y**2          # x = 0
    ]
        
    btype = [
        'dirichlet',
        'dirichlet',
        'dirichlet',
        'dirichlet'        
    ]
    
    return f, g, btype, u_exact

def example_6(eig_1=None,eig_2=None,angle=None,Amp=1.0,modes=None,L=1.0):
    """
    Linear–quadratic manufactured solution for anisotropic diffusion.

    Exact solution:
        .. math::
            u(x,y) = 2x + y^2.

    PDE forcing:
        .. math::
            f = 2 A_{22}.

    Boundary conditions:
        Dirichlet on all sides.

    Returns
    -------
    f : callable
        Right-hand side :math:`f(x,y)`.
    g : list of callables
        Dirichlet boundary data in the order [y=0, x=L, y=L, x=0].
    btype : list of str
        Boundary condition types (all Dirichlet).
    u_exact : callable
        Exact solution :math:`u(x,y)`.
    """
    
    def u_exact(p,A):
        x,y = p
        return 2*x+y**2
    
    def f(p,A):
        return 2*A[1,1]
    
    g = [
        lambda x: 2*x,      #y=0
        lambda y: 2*L+y**2, #x=L
        lambda x: 2*x+L**2, #y=L
        lambda y: y**2      #x=0
    ]
    
    btype = [
        'dirichlet',
        'dirichlet',
        'dirichlet',
        'dirichlet'        
    ]
    
    return f, g, btype, u_exact

def example_7(eig_1=None,eig_2=None,angle=None,Amp=1.0,modes=None,L=1.0):
    """
    Same manufactured solution as Example 6 with mixed boundary conditions.

    Exact solution:
        .. math::
            u(x,y) = 2x + y^2.

    Boundary conditions:
        Dirichlet on y=0 and x=0,
        Neumann on x=L and y=L.

    Returns
    -------
    f : callable
        Right-hand side :math:`f(x,y)`.
    g : list of callables
        Dirichlet boundary data in the order [y=0, x=L, y=L, x=0].
    btype : list of str
        Boundary condition types.
    u_exact : callable
        Exact solution :math:`u(x,y)`.
    """
    
    def u_exact(p,A):
        x,y = p
        return 2*x+y**2
    
    def f(p,A):
        return 2*A[1,1]
    
    def A11(x,y):
        return (eig_1(np.array([x,y]))*(np.cos(angle(np.array([x,y])))**2)+
                eig_2(np.array([x,y]))*(np.sin(angle(np.array([x,y])))**2))

    def A12(x,y):
        delta_eig = eig_1(np.array([x,y]))-eig_2(np.array([x,y]))
        return (delta_eig*np.sin(angle(np.array([x,y])))*
                          np.cos(angle(np.array([x,y]))))

    def A22(x,y):
        return (eig_2(np.array([x,y]))*(np.cos(angle(np.array([x,y])))**2)+
                eig_1(np.array([x,y]))*(np.sin(angle(np.array([x,y])))**2))
    
    g = [
        lambda x: 2*x,      #y=0
        lambda y: 2*L,      #x=L
        lambda x: 2,        #y=L
        lambda y: y**2      #x=0
    ]
    
    btype = [
        'dirichlet',
        'neumann',
        'neumann',
        'dirichlet'        
    ]
    
    return f, g, btype, u_exact

def example_8(eig_1,eig_2,angle,Amp=1.0,modes=None,L=1.0):
    """
    Exponential manufactured solution for anisotropic diffusion.

    Exact solution:
        .. math::
            u(x,y) = e^{x^2 + y^2}.

    Forcing:
        .. math::
            f = (2\,\\mathrm{trace}(A) + 4 p^T A p)\, e^{\\|p\\|^2}.

    Boundary conditions:
        Mixed Dirichlet/Neumann.
     
    Returns
    -------
    f : callable
        Right-hand side :math:`f(x,y)`.
    g : list of callables
        Dirichlet boundary data in the order [y=0, x=L, y=L, x=0].
    btype : list of str
        Boundary condition types.
    u_exact : callable
        Exact solution :math:`u(x,y)`.
    """
    
    def u_exact(p,A):
        return np.exp(np.linalg.norm(p,axis=0)**2)
    
    def f(p,A):
        return ((2*np.trace(A)+4*np.dot(p, A @p))
                *np.exp(np.linalg.norm(p,axis=0)**2))
    
    def A11(x,y):
        return (eig_1(np.array([x,y]))*(np.cos(angle(np.array([x,y])))**2)+
                eig_2(np.array([x,y]))*(np.sin(angle(np.array([x,y])))**2))

    def A12(x,y):
        delta_eig = eig_1(np.array([x,y]))-eig_2(np.array([x,y]))
        return (delta_eig*np.sin(angle(np.array([x,y])))*
                          np.cos(angle(np.array([x,y]))))

    def A22(x,y):
        return (eig_2(np.array([x,y]))*(np.cos(angle(np.array([x,y])))**2)+
                eig_1(np.array([x,y]))*(np.sin(angle(np.array([x,y])))**2))

    g = [
        lambda x: np.exp(x**2),                                     # y=0
        lambda y: 2*np.exp(L**2+y**2)*(L*A11(L,y)+y*A12(L,y)),      # x=L
        lambda x: np.exp(x**2+L**2),                                # y=L
        lambda y: -2*y*A12(0.0,y)*np.exp(y**2)                      # x=0
    ]
    
    btype = [
        'dirichlet',
        'neumann',
        'dirichlet',
        'neumann'
    ]
    
    return f, g, btype, u_exact

def example_9(eig_1,eig_2,angle,Amp=1.0,modes=None,L=1.0):
    """
    Boundary-layer manufactured solution for anisotropic diffusion.

    Exact solution:
        .. math::
            u(x,y) = \\tanh\\left(\\frac{y - L/2}{\\sqrt{A_{22}}}\\right).

    Forcing:
        .. math::
            f = -2\\tanh(z)/\\cosh^2(z).

    Boundary conditions:
        Dirichlet on y=0,L; 0-Neumann on x=0,L.

    Returns
    -------
    f : callable
        Right-hand side :math:`f(x,y)`.
    g : list of callables
        Dirichlet boundary data in the order [y=0, x=L, y=L, x=0].
    btype : list of str
        Boundary condition types.
    u_exact : callable
        Exact solution :math:`u(x,y)`.
    """
    
    def u_exact(p,A):
        x,y = p
        z = (y-1/2*L)/np.sqrt(A[1,1])
        return np.tanh(z)
    
    def f(p,A):
        x,y = p
        z = (y-1/2*L)/np.sqrt(A[1,1])
        return - 2 *np.tanh(z) / np.cosh(z)**2
    
    def A11(x,y):
        return (eig_1(np.array([x,y]))*(np.cos(angle(np.array([x,y])))**2)+
                eig_2(np.array([x,y]))*(np.sin(angle(np.array([x,y])))**2))

    def A12(x,y):
        delta_eig = eig_1(np.array([x,y]))-eig_2(np.array([x,y]))
        return (delta_eig*np.sin(angle(np.array([x,y])))*
                          np.cos(angle(np.array([x,y]))))

    def A22(x,y):
        return (eig_2(np.array([x,y]))*(np.cos(angle(np.array([x,y])))**2)+
                eig_1(np.array([x,y]))*(np.sin(angle(np.array([x,y])))**2))

    def _q(x, y, sign):
        A22v = A22(x,y)
        z = (y-0.5*L)/np.sqrt(A22v)
        return sign * A12(x,y) / (np.cosh(z)**2 * np.sqrt(A22v))

    g = [
        lambda x: np.tanh(-0.5*L/np.sqrt(A22(x,0.0))),  # y=0  Dirichlet
        lambda y: _q(L, y, +1.0),                        # x=L  Neumann
        lambda x: np.tanh(0.5*L/np.sqrt(A22(x,L))),      # y=L  Dirichlet
        lambda y: _q(0.0, y, -1.0)                        # x=0  Neumann
    ]
    
    btype = [
        'dirichlet',
        'neumann',
        'dirichlet',
        'neumann'
    ]

    return f, g, btype, u_exact

def example_10(eig_1,eig_2,angle,Amp=1.0,modes=None,L=1.0):
    """
    Spike-like manufactured solution with anisotropy and mixed BCs.

    Exact solution:
        .. math::
            u(x,y) = c(y)*a(x)*b(x),

        where a(x) are Gaussian bumps, b(x)=x^2*(L-x)^2,
        and c(y)=2*Amp*sqrt(y+L/2).

    Boundary conditions:
        Dirichlet on y=0,L; 0-Neumann on x=0,L.

    Returns
    -------
    f : callable
        Right-hand side :math:`f(x,y)`.
    g : list of callables
        Dirichlet boundary data in the order [y=0, x=L, y=L, x=0].
    btype : list of str
        Boundary condition types.
    u_exact : callable
        Exact solution :math:`u(x,y)`.
    """
    
    alpha = 0.01*L
    beta  = L-alpha
        
    def a(x,delta):
        return (np.exp(-(x-alpha)**2/delta)+
                np.exp(-(x- beta)**2/delta))
    
    def a_d(x,delta):
        return (-2/delta*((x-alpha)*np.exp(-(x-alpha)**2/delta)+
                       (x- beta)*np.exp(-(x- beta)**2/delta)))
    
    def a_dd(x,delta):
        return (-2/delta*a(x, delta)+4/delta**2*
                ((x-alpha)**2*np.exp(-(x-alpha)**2/delta)+
                 (x- beta)**2*np.exp(-(x- beta)**2/delta)))
    
    def b(x):
        return x**2*(L-x)**2
    
    def b_d(x):
        return 2*x*(L**2-3*x*L+2*x**2)
    
    def b_dd(x):
        return 2*L**2-12*x*L+12*x**2
    
    def c(y):
        return 2.0*Amp*np.sqrt(y+L/2)
    
    def c_d(y):
        return 1.0*Amp/np.sqrt(y+L/2)
    
    def c_dd(y):
        return -1.0*Amp/2/np.sqrt(y+L/2)**3
        
    def u_xx(p,delta):
        x,y = p
        return c(y)*(a_dd(x,delta)*b(x)+2*a_d(x,delta)*b_d(x)+a(x,delta)*b_dd(x))
    
    def u_xy(p,delta):
        x,y = p
        return c_d(y)*(a_d(x,delta)*b(x)+a(x,delta)*b_d(x))
    
    def u_yy(p,delta):
        x,y = p
        return c_dd(y)*a(x,delta)*b(x)
    
    def u_exact(p,A):
        x,y = p
        delta = A[0,0]
        return c(y)*a(x,delta)*b(x)
    
    def f(p,A):
        delta = A[0,0]
        return delta*u_xx(p,delta)+2*A[0,1]*u_xy(p,delta)+A[1,1]*u_yy(p,delta)
    
    def A11(x,y):
        return (eig_1(np.array([x,y]))*(np.cos(angle(np.array([x,y])))**2)+
                eig_2(np.array([x,y]))*(np.sin(angle(np.array([x,y])))**2))

    def A12(x,y):
        delta_eig = eig_1(np.array([x,y]))-eig_2(np.array([x,y]))
        return (delta_eig*np.sin(angle(np.array([x,y])))*
                          np.cos(angle(np.array([x,y]))))

    def A22(x,y):
        return (eig_2(np.array([x,y]))*(np.cos(angle(np.array([x,y])))**2)+
                eig_1(np.array([x,y]))*(np.sin(angle(np.array([x,y])))**2))

    g = [
        lambda x: c(0.0)*a(x,A11(x,0.0))*b(x), #y=0
        lambda y: 0.0,                         #x=L
        lambda x: c(L)*a(x,A11(x,L))*b(x),     #y=L
        lambda y: 0.0                          #x=0
    ]
    
    btype = [
        'dirichlet',
        'neumann',
        'dirichlet',
        'neumann'
    ]

    return f, g, btype, u_exact

# Radial Domain
# Solution with 0-Dritichlet and forcing term (no nodes, matrices)
def example_0(eig_1=None,eig_2=None,angle=None,Amp=1.0,modes=None,L=1.0):
    """
    Poisson problem on the unit disk with zero Dirichlet boundary conditions.

    PDE:
        .. math::
            \\Delta u = -2\\pi^2 \sin(\\pi x)\sin(\\pi y).

    Exact solution:
        Particular solution minus a Fourier-series homogeneous correction
        expressed in polar coordinates.

    Boundary conditions:
        Dirichlet: u = 0 on r = R.

    Returns
    -------
    f : callable
        Right-hand side :math:`f(x,y)`.
    g : list of callables
        Dirichlet boundary data in the order [y=0, x=1, y=1, x=0].
    btype : list of str
        Boundary condition types (all Dirichlet).
    u_exact : callable
        Exact solution :math:`u(x,y)`.
    """
    
    def u_exact(p, max_modes=12):
        """
        Corrected analytical solution for Delta u = -2pi^2 sin(pi x) sin(pi y)
        on a unit circle with 0-Dirichlet using a Sine Fourier Expansion.
        """
        X_nodes, Y_nodes = p
        # Convert Cartesian input nodes to Polar coordinates
        R_nodes = np.sqrt(X_nodes**2 + Y_nodes**2)
        Theta_nodes = np.arctan2(Y_nodes, X_nodes)
        
        # 1. Particular Solution Component: Up = sin(pi x) * sin(pi y)
        u_particular = np.sin(np.pi * X_nodes) * np.sin(np.pi * Y_nodes)
        
        # 2. Homogeneous Correction Component: Uh = Sum (B_n*r^n*sin(n*theta))
        u_homogeneous = np.zeros_like(X_nodes)
        
        # Pre-calculate Fourier coefficients B_n via numerical integration
        for n in range(1, max_modes + 1):
            # Boundary integrand matching the odd symmetry profile
            integrand = lambda t: (np.sin(np.pi * np.cos(t)) *
                                   np.sin(np.pi * np.sin(t)) * np.sin(n * t))
            
            # Integrate from 0 to 2*pi
            coeff, _ = integrate.quad(integrand, 0, 2 * np.pi)
            B_n = coeff / np.pi
                
            # Add mode contribution to all points: B_n * r^n * sin(n * theta)
            u_homogeneous += B_n * (R_nodes**n) * np.sin(n * Theta_nodes)
            
        # Total Solution = Particular - Homogeneous Boundary Correction
        return u_particular - u_homogeneous
    
    # RHS function
    def f(p):
        x,y = p
        
        return - 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def g(p):
        return 0.0
    
    btype = [
        'dirichlet'       
    ]
    
    return f, g, btype, u_exact