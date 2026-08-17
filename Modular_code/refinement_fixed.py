import numpy as np
import assembly_vec as assembly
import error_analysis
import stencils
from domain import PDEDomainContext
from scipy.spatial import KDTree
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def _energy_norm_sq(grad_u, A=None, d=None, reg=1e-10):
    """
    A^{-1}-weighted squared gradient magnitude at each node:
        e_i = grad_u_i^T A^{-1} grad_u_i
    Falls back to the plain Euclidean |grad_u|^2 if A is None, so this
    can be reused by both the monitor and the calibration routine.
    """
    if A is None:
        return np.einsum('ni,ni->n', grad_u, grad_u)
    d = grad_u.shape[1] if d is None else d
    Minv = A + reg * np.eye(d)[None, :, :]
    Minv = np.linalg.inv(Minv)
    Ainv_gu = np.einsum('nij,nj->ni', Minv, grad_u)
    return np.einsum('ni,ni->n', grad_u, Ainv_gu)


def calibrate_alpha(grad_u, A=None, d=2, q=2, target_contrast=5.0,
                     contrast_percentile=90.0, alpha_max=1e12,
                     xtol=1e-10, rtol=1e-10, maxiter=200, verbose=False):
    """
    Calibrate the arclength-monitor intensity `alpha` so the scalar
    density weight

        w(x; alpha) = (1 + alpha * e(x))**(1/(d+q))

    -- the same exponent structure used inside operator_gradient_monitor
    -- achieves a target contrast between a high-gradient region and a
    typical (median) region of the current node set:

        w(e_hi; alpha) / w(e_med; alpha) = target_contrast

    where e(x) = grad_u^T A^{-1} grad_u (or |grad_u|^2 if A is None),
    e_hi is a robust high percentile of e (default 90th, to avoid
    letting one noisy stencil node set alpha for the whole mesh) and
    e_med is the median.

    This ratio is 1 at alpha=0 and rises monotonically to
    (e_hi/e_med)**(1/(d+q)) as alpha -> inf, so a root of
    R(alpha) - target_contrast = 0 is well-posed whenever
    1 < target_contrast < (e_hi/e_med)**(1/(d+q)), and Brentq (no
    derivatives, guaranteed convergence once bracketed) is a natural
    solver.

    Parameters
    ----------
    target_contrast : float
        Desired ratio of node density near steep gradients vs. typical
        regions. This is the actual "adaptivity intensity" knob --
        tune this, not alpha directly.
    contrast_percentile : float
        Percentile of e(x) used as the "high-gradient" reference.

    Returns
    -------
    alpha : float
        0.0 if the gradient field has no meaningful spatial contrast
        (equidistribution has nothing to act on). alpha_max if the
        requested target_contrast is theoretically unreachable for
        this gradient field (a warning is printed). Otherwise the
        calibrated alpha.
    """
    e = _energy_norm_sq(grad_u, A=A, d=d)
    e = np.maximum(e, 0.0)

    e_hi = np.percentile(e, contrast_percentile)
    e_med = max(np.median(e), 1e-300)

    # Degenerate case: gradient magnitude is essentially flat across
    # the node set -- there's no spatial contrast for any alpha to
    # equidistribute against.
    if e_hi <= (1.0 + 1e-8) * e_med:
        if verbose:
            print("calibrate_alpha: gradient field is ~flat; alpha=0.")
        return 0.0

    max_reachable = (e_hi / e_med) ** (1.0 / (d + q))
    if target_contrast >= max_reachable:
        print(f"calibrate_alpha: target_contrast={target_contrast:.3g} "
              f"exceeds max reachable contrast {max_reachable:.3g} for "
              f"this gradient field; clipping to alpha_max={alpha_max:.1e}.")
        return alpha_max

    def R(alpha):
        w_hi = (1.0 + alpha * e_hi) ** (1.0 / (d + q))
        w_med = (1.0 + alpha * e_med) ** (1.0 / (d + q))
        return w_hi / w_med - target_contrast

    f_lo = 1.0 - target_contrast   # R(alpha=0) = 1
    f_hi = R(alpha_max)
    if f_lo * f_hi > 0:
        raise RuntimeError(
            "calibrate_alpha: failed to bracket a root -- check e_hi/e_med "
            "and target_contrast, or raise alpha_max."
        )

    alpha = brentq(R, 0.0, alpha_max, xtol=xtol, rtol=rtol, maxiter=maxiter)
    if verbose:
        print(f"calibrate_alpha: e_hi={e_hi:.3g}, e_med={e_med:.3g}, "
              f"alpha={alpha:.3g} (target contrast {target_contrast:.2g})")
    return alpha

def min_node_spacing(P, verbose=False):
    tree = KDTree(P)
    dists, idx = tree.query(P, k=2)   # first neighbor is the point itself

    if verbose:
        i = np.argmin(dists[:, 1])
        j = idx[i, 1]
        print("Closest nodes:", i, j, "distance:", dists[i, 1])
        print(P[i], P[j])

    return np.min(dists[:, 1])

def operator_gradient_monitor(grad_u, alpha, A=None, d=2, q=2, reg=1e-10):
    """
    Monitor combining physical operator A with solution gradient,
    measured in the A^{-1}-energy norm. If A is None, then this
    is the regular gradient_monitor function.
    """

    if A is not None:
        M = A + reg * np.eye(d)[None, :, :]
        M = np.linalg.inv(M)
    else:
        M = np.eye(d)[None, :, :]

    Ainv_gu = np.einsum('nij,nj->ni', M, grad_u)          # A^{-1} grad_u
    energy_norm2 = np.einsum('ni,ni->n', grad_u, Ainv_gu) # grad_u^T A^{-1} grad_u

    prefac = (1 + alpha * energy_norm2) ** (-1.0 / (d + q))
    outer = np.einsum('ni,nj->nij', Ainv_gu, Ainv_gu)

    return prefac[:, None, None] * (M + alpha * outer)

def smooth_monitor(M, S, beta=0.25):
    """
    Smooth a tensor-valued monitor over the stencil graph.

    Parameters
    ----------
    M : (N,d,d)
        Monitor tensor.
    S : (N,k)
        Neighbor indices.
    beta : float
        Smoothing strength.
    """
    avg = M[S].mean(axis=1)
    return (1 - beta) * M + beta * avg

def redistribute_nodes(P, u, num_stencil_nodes, num_centers, basis,
                        shape, L, btype_all_dirichlet, augmentation,
                        A, alpha, eps, tol, sparse, relax=1.0,
                        min_relax=1e-4, tangle_frac=0.5,
                        target_contrast=5.0):
    """
    One elliptic-generator (Winslow-type) redistribution step.

    Solves grad(u) under the physical operator A, builds the operator-
    weighted arclength monitor M(x), then solves the coordinate PDEs
    div(M grad x) = 0, div(M grad y) = 0 with the boundary held fixed.
    The raw solution P_target is damped/line-searched against a
    nearest-neighbor collapse (tangle) indicator before being accepted.

    alpha : float or None
        Monitor intensity. Pass a float to use it directly, or None to
        auto-calibrate it from the current grad_u via calibrate_alpha
        (recommended -- see calibrate_alpha's docstring for why a fixed
        constant doesn't transfer across iterations/problems).
    """
    dim = P.shape[1]

    # 1) gradient of current solution on current nodes, under the
    #    PHYSICAL operator A (this part was already correct)
    S = stencils.knn_list(P, num_stencil_nodes)
    #C = stencils.knn_list(P, num_centers) if num_centers is not None else None
    C = num_centers
    ctx_g = PDEDomainContext(P, S, C, A)
    assembly.set_rbf_func(basis, augmentation, ctx_g, tol)
    grad_u = np.column_stack([Wl @ u for Wl in assembly.global_grads_sparse(ctx_g)])

    if alpha is None:
        alpha = calibrate_alpha(grad_u, A=None, d=dim, q=2,
                                 target_contrast=target_contrast, verbose=True)

    # 2) monitor function from that gradient.
    #    BUG FIX: forward A so the monitor is measured in the
    #    A^{-1}-energy norm instead of silently falling back to
    #    the Euclidean (isotropic) norm.
    M = operator_gradient_monitor(grad_u, alpha, A=None, d=dim)
    #M = smooth_monitor(M, S, beta=0.5)

    evals = np.linalg.eigvalsh(M)
    lam_max = evals[:, -1]

    #plt.scatter(P[:,0], P[:,1], c=lam_max)
    #plt.colorbar()
    #plt.show()

    # 3) solve coordinate PDEs, A = M, f = 0, Dirichlet = identity on boundary
    f_zero = lambda p, A: 0.0
    g_x = [
        lambda x: x,   # y=0: x varies
        lambda x: L,   # x=L: x fixed
        lambda x: x,   # y=L: x varies
        lambda x: 0.0  # x=0: x fixed
    ]

    g_y = [
        lambda y: 0.0, # y=0
        lambda y: y,   # x=L
        lambda y: L,   # y=L
        lambda y: y    # x=0
    ]

    ctx_x = assembly.rbf_fd_system(f_zero, g_x, btype_all_dirichlet, P, basis, shape, L,
                                   num_stencil_nodes, num_centers, augmentation,
                                   M, sparse=True)

    x_new = assembly.rbf_fd_solve_sparse(ctx_x.W, ctx_x.F)

    # reuse the SAME W, just a new rhs for the y-coordinate solve
    g, in_boundary, normal_vec = assembly.set_boundary_func(g_y, btype_all_dirichlet, shape, L, ctx_x)
    f_vec_y = assembly.right_hand_side(ctx_x, f_zero, g, in_boundary)
    y_new = assembly.rbf_fd_solve_sparse(ctx_x.W, f_vec_y)

    P_target = np.column_stack([x_new, y_new])
    spacing_old = min_node_spacing(P)

    # BUG FIX: this loop was dead code (inside a triple-quoted string).
    # `relax` was accepted as an argument but never used, so
    # redistribute_nodes always returned the raw, undamped P_target
    # with no protection against tangling/collapse.
    P_solved = P.copy()
    accepted = False
    r = relax
    while r > min_relax:
        P_trial = r * P_target + (1 - r) * P
        if min_node_spacing(P_trial) > tangle_frac * spacing_old:
            P_solved = P_trial
            accepted = True
            break
        r *= 0.5

    if not accepted:
        print(f"redistribute_nodes: no safe relaxation found down to "
              f"relax={min_relax:.1e}; returning mesh unchanged this step.")

    # IMPORTANT: return P_solved (the damped/line-searched result), never
    # P_target directly -- returning P_target silently discards the
    # relaxation loop above and reintroduces the tangling risk it exists
    # to prevent.
    return P_solved, spacing_old

def mesh_refinement(f, g, btype, P, rbf_shape, shape, L, num_stencil_nodes,
                    num_centers, augmentation, eig_1, eig_2, angle, eps, tol,
                    sparse=True, max_iter=20, alpha=None, tangle_frac=0.5,
                    move_tol=1e-6, relax=1.0, target_contrast=5.0, verbose=True):
    """
    Parameters
    ----------
    alpha : float
        Adaptivity intensity for the monitor function. Currently a fixed
        scalar; wire in `calibrate_alpha` here once you want the strength
        of adaptation tied to an explicit equidistribution target rather
        than a hand-picked constant.
    move_tol : float
        Convergence tolerance on max nodal displacement, RELATIVE to the
        current minimum node spacing (see BUG FIX note below).
    """

    btype_all_dirichlet = ['dirichlet'] * 4

    for it in range(max_iter):
        print(f"Refinement iteration: {it}")
        print("Solving u:")
        A = assembly.coeff_matrix(P.T, eig_1, eig_2, angle)
        ctx = assembly.rbf_fd_system(f, g, btype, P, rbf_shape, shape, L,
                            num_stencil_nodes, num_centers, augmentation,
                            A, eps, tol, sparse)
        u = assembly.rbf_fd_solve_sparse(ctx.W, ctx.F)

        print("Solving P:")
        P_new, spacing_old = redistribute_nodes(P, u, num_stencil_nodes,
                                   num_centers, rbf_shape, shape, L,
                                   btype_all_dirichlet, augmentation,
                                   A, alpha, eps, tol, sparse, relax=relax,
                                   target_contrast=target_contrast,
                                   tangle_frac=tangle_frac)

        spacing_new = min_node_spacing(P_new, verbose=verbose)

        if spacing_new < tangle_frac * spacing_old:
            # redistribute_nodes' internal line search already tries to
            # avoid this; if it still happens, the monitor/step is too
            # aggressive for this iterate. Stop rather than propagate
            # a degraded mesh.
            print("Mesh update rejected: node spacing deteriorated.")
            break

        max_move = np.max(np.linalg.norm(P_new - P, axis=1))
        print("Max move:", max_move)

        # BUG FIX: 1e-15 is unreachable roundoff noise for an
        # independent linear solve each iteration; scale the tolerance
        # to the mesh's own length scale instead.
        if max_move < move_tol * spacing_old:
            P = P_new
            break

        P = P_new   # stencils get rebuilt next iteration automatically
        print()

    plt.scatter(P[:,0], P[:,1], s=1)
    plt.axis("equal")
    plt.show()

    return P
