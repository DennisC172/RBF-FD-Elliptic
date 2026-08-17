import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import examples


def build_tensor(eig_1, eig_2, angle):
    """A = R(angle) diag(eig_1, eig_2) R(angle)^T -- SPD anisotropy tensor."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    D = np.diag([eig_1, eig_2])
    return R @ D @ R.T


def solve_fd(n, A, f_fn, u_exact_fn, L=1.0, return_matrix=False):
    """
    Standard 2nd-order central-difference discretization of
        A11 u_xx + 2 A12 u_xy + A22 u_yy = f
    on an n x n uniform grid over [0,L]^2 (n points per side, h = L/(n-1)),
    with homogeneous Dirichlet BCs.

    Returns dict with grids, computed solution, exact solution, timings,
    and both requested error metrics.
    """
    t0 = time.time()
    h = L / (n - 1)
    xs = np.linspace(0, L, n)
    ys = np.linspace(0, L, n)
    X, Y = np.meshgrid(xs, ys, indexing='ij')  # X[i,j]=xs[i], Y[i,j]=ys[j]

    A11, A12, A22 = A[0, 0], A[0, 1], A[1, 1]
    cxx = A11 / h**2
    cyy = A22 / h**2
    cxy = A12 / (2 * h**2)
    ccenter = -2 * cxx - 2 * cyy

    IDX = np.arange(n * n).reshape(n, n)

    i_int = np.arange(1, n - 1)
    j_int = np.arange(1, n - 1)
    II, JJ = np.meshgrid(i_int, j_int, indexing='ij')
    row_idx = IDX[II, JJ].ravel()

    rows_list, cols_list, vals_list = [], [], []

    def add_stencil(oi, oj, coeff):
        col_idx = IDX[II + oi, JJ + oj].ravel()
        rows_list.append(row_idx)
        cols_list.append(col_idx)
        vals_list.append(np.full(row_idx.shape, coeff))

    add_stencil(0, 0, ccenter)
    add_stencil(1, 0, cxx)
    add_stencil(-1, 0, cxx)
    add_stencil(0, 1, cyy)
    add_stencil(0, -1, cyy)
    add_stencil(1, 1, cxy)
    add_stencil(-1, -1, cxy)
    add_stencil(1, -1, -cxy)
    add_stencil(-1, 1, -cxy)

    F = f_fn((X, Y), A)
    b = np.zeros(n * n)
    b[row_idx] = F[II, JJ].ravel()

    boundary_mask = np.ones((n, n), dtype=bool)
    boundary_mask[1:-1, 1:-1] = False
    b_idx = IDX[boundary_mask]
    rows_list.append(b_idx)
    cols_list.append(b_idx)
    vals_list.append(np.ones_like(b_idx, dtype=float))
    # g = 0 on boundary -> b already 0 there

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    Amat = sp.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n))
    t_assemble = time.time() - t0

    t1 = time.time()
    u = spla.spsolve(Amat, b)
    t_solve = time.time() - t1
    U = u.reshape(n, n)

    Uex = u_exact_fn((X, Y), A)

    # --- solution error: u - u_exact ---
    e = U - Uex
    e_linf = np.max(np.abs(e))/np.max(Uex)   
    e_l2 = np.sqrt(np.sum(e**2))/np.linalg.norm(Uex)   

    # --- truncation/consistency error: L_h(u_exact) - f, interior points only ---
    Lh_uex_int = (
        cxx * (Uex[2:, 1:-1] - 2 * Uex[1:-1, 1:-1] + Uex[:-2, 1:-1])
        + cyy * (Uex[1:-1, 2:] - 2 * Uex[1:-1, 1:-1] + Uex[1:-1, :-2])
        + cxy * (Uex[2:, 2:] - Uex[2:, :-2] - Uex[:-2, 2:] + Uex[:-2, :-2])
    )
    r = Lh_uex_int - F[1:-1, 1:-1]
    r_linf = np.max(np.abs(r))/np.max(np.abs(F[1:-1, 1:-1]))
    r_l2 = np.sqrt(np.sum(r**2))/np.linalg.norm(F[1:-1, 1:-1])   

    result = dict(
        n=n, h=h, nnz=Amat.nnz, t_assemble=t_assemble, t_solve=t_solve,
        e_linf=e_linf, e_l2=e_l2, r_linf=r_linf, r_l2=r_l2,
        U=U, Uex=Uex, X=X, Y=Y,
    )
    if return_matrix:
        result['Amat'] = Amat
    return result


if __name__ == "__main__":
    L = 1.0
    Amp = 1.0
    modes = [1.0, 1.0]
    eig_1_str = "lambda p: 1e0"
    eig_2_str = "lambda p: 1e-4"
    angle_str = "lambda p: 12.0/24.0*np.pi"
    print(f'Eig_1 = {eig_1_str}')
    print(f'Eig_2 = {eig_2_str}')
    print(f'Angle = {angle_str}')
    
    eig_1 = eval(eig_1_str)
    eig_2 = eval(eig_2_str)
    angle = eval(angle_str)
    A = build_tensor(eig_1(0), eig_2(0), angle(0))
    f_fn, g, btype, u_exact_fn = examples.example_11(eig_1, eig_2, angle, Amp=Amp, modes=modes, L=L)

    print("A =\n", A)
    print(f"{'n':>5} {'h':>12} {'||e||_inf':>14} {'||e||_2':>14} {'||r||_inf':>14} {'||r||_2':>14}")
    prev = None
    for n in [25, 50, 100, 700]:
        res = solve_fd(n, A, f_fn, u_exact_fn, L=L)
        print(f"{n:5d} {res['h']:12.6f} {res['e_linf']:14.6e} {res['e_l2']:14.6e} "
              f"{res['r_linf']:14.6e} {res['r_l2']:14.6e}")
        if prev is not None:
            rate_e = np.log2(prev['e_linf'] / res['e_linf'])
            rate_r = np.log2(prev['r_linf'] / res['r_linf'])
            print(f"      observed order (Linf):  solution={rate_e:.3f}   truncation={rate_r:.3f}")
        prev = res
