"""Operator Inference on a (polynomial) manifold: plain latent and geometric
manifold-Galerkin reduced models.

Implements the GMG-OpInf method.  Given a manifold

    Gamma(shat) = s_ref + V shat + Vbar p(shat),
    J_Gamma(shat) = V + Vbar J_p(shat),        V^T V = I,  V^T Vbar = 0,

two non-intrusive reduced models are provided for a full-order model
``s' = f(s; mu)``:

* **Plain latent OpInf** (Geelen-style): ``shat' = c + A shat + N p(shat)``,
  the exact reduced dynamics of an on-manifold state projected through the
  linear encoder.  Structurally fragile: the latent ODE discards the manifold
  metric, so integration drift leaves the certified collar and the polynomial
  features blow up.

* **Geometric manifold-Galerkin (GMG) OpInf**: solve the tangent condition
  ``J_Gamma shat' = f`` in least squares, giving the mass-weighted system

      shat' = M(shat)^{-1} ( a(shat) + J_p(shat)^T b(shat) ),
      M(shat) = I + J_p^T (Vbar^T Vbar) J_p,   1 <= lambda(M) <= L^2,

  with ``a = V^T f`` and ``b = Vbar^T f`` learned exactly like the plain
  operators.  The manifold's Lipschitz budget conditions the mass matrix, so
  the same certificate that bounds the decoder stabilizes the dynamics.

The module also carries the shared non-intrusive machinery: the fourth-order
forward/backward/central time-derivative stencil (FBCT4), the order-wise
Tikhonov operator regression, RK4 / semi-implicit rollouts, and the entrywise
spline interpolation (cubic / quartic / quintic) plus affine extrapolation of
per-parameter operator sets.

Feature maps are supplied as callables ``feature_fn(s) -> p`` and
``feature_jac_fn(s) -> (p, J_p)``; :func:`opm_feature_fn` /
:func:`opm_feature_jac_fn` build them for a fitted :class:`~greedyopm.GreedyOPM`
from the differentiated Arnoldi recurrence (the exact gradients the certificate
uses).  Monomial-manifold baselines can pass any JAX-differentiable feature map.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

FeatureFn = Callable[[jnp.ndarray], jnp.ndarray]  # (r,) -> (q,) nonlinear feats

# ===========================================================================
# FBCT4: fourth-order forward/backward/central time derivatives
# ===========================================================================

_MIN_SAMPLES = 5


def fbct4(X: np.ndarray, dt: float) -> np.ndarray:
    """Fourth-order FBCT time derivative of a single trajectory.

    Arguments
    ---------
    X : (d, N)
        State (or reduced-coordinate) trajectory, columns ordered in time on a
        uniform grid of spacing ``dt``.  ``N >= 5`` is required.
    dt : float
        Uniform time-step.

    Returns
    -------
    dX : (d, N)
        Column ``j`` is the fourth-order estimate of ``dX/dt`` at time index
        ``j``; forward at ``j in {0, 1}``, central for ``2 <= j <= N-3``,
        backward at ``j in {N-2, N-1}``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (d, N); got shape {X.shape}.")
    d, N = X.shape
    if N < _MIN_SAMPLES:
        raise ValueError(
            f"FBCT4 needs at least {_MIN_SAMPLES} time samples; got N={N}."
        )
    dX = np.empty_like(X)
    twelve_dt = 12.0 * dt

    # -- first two points: fourth-order forward stencils --------------------
    dX[:, 0] = (
        -25.0 * X[:, 0] + 48.0 * X[:, 1] - 36.0 * X[:, 2]
        + 16.0 * X[:, 3] - 3.0 * X[:, 4]
    ) / twelve_dt
    dX[:, 1] = (
        -3.0 * X[:, 0] - 10.0 * X[:, 1] + 18.0 * X[:, 2]
        - 6.0 * X[:, 3] + 1.0 * X[:, 4]
    ) / twelve_dt

    # -- interior: fourth-order central stencil -----------------------------
    # dX[:, j] = (X[j-2] - 8 X[j-1] + 8 X[j+1] - X[j+2]) / (12 dt), 2<=j<=N-3
    dX[:, 2:N - 2] = (
        X[:, 0:N - 4] - 8.0 * X[:, 1:N - 3]
        + 8.0 * X[:, 3:N - 1] - X[:, 4:N]
    ) / twelve_dt

    # -- last two points: fourth-order backward stencils --------------------
    dX[:, N - 2] = (
        -1.0 * X[:, N - 5] + 6.0 * X[:, N - 4] - 18.0 * X[:, N - 3]
        + 10.0 * X[:, N - 2] + 3.0 * X[:, N - 1]
    ) / twelve_dt
    dX[:, N - 1] = (
        3.0 * X[:, N - 5] - 16.0 * X[:, N - 4] + 36.0 * X[:, N - 3]
        - 48.0 * X[:, N - 2] + 25.0 * X[:, N - 1]
    ) / twelve_dt
    return dX


def fbct4_blocked(X: np.ndarray, dt: float, n_per: int) -> np.ndarray:
    """FBCT4 derivative of several equal-length trajectories laid side by side.

    ``X`` is a ``(d, n_traj * n_per)`` matrix whose columns are ``n_traj``
    consecutive trajectories of ``n_per`` samples each (the layout produced by
    concatenating per-parameter snapshot blocks).  Each block is differentiated
    independently so the stencils never straddle a parameter boundary.

    Arguments
    ---------
    X : (d, n_traj * n_per)
    dt : float
    n_per : int
        Samples per trajectory.

    Returns
    -------
    dX : (d, n_traj * n_per)
    """
    X = np.asarray(X, dtype=float)
    d, k = X.shape
    if k % n_per != 0:
        raise ValueError(
            f"Column count {k} is not a multiple of n_per={n_per}."
        )
    n_traj = k // n_per
    dX = np.empty_like(X)
    for b in range(n_traj):
        sl = slice(b * n_per, (b + 1) * n_per)
        dX[:, sl] = fbct4(X[:, sl], dt)
    return dX

# ===========================================================================
# Operator Inference regression
# ===========================================================================


def infer_operators(
    Shat: np.ndarray,
    dShat: np.ndarray,
    P_nl: np.ndarray | None,
    *,
    lam_lin: float = 0.0,
    lam_nl: float = 0.0,
    lam_const: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Least-squares fit of ``dShat = c + A Shat + N P_nl`` (order-wise ridge).

    Arguments
    ---------
    Shat : (r, k)   reduced coordinates.
    dShat : (r, k)  reduced time derivatives (FBCT4 estimate).
    P_nl : (q, k) or None
        Active nonlinear feature values on the same coordinates; ``None`` or an
        empty array yields a linear (constant + linear) model.
    lam_lin, lam_nl, lam_const : float
        Tikhonov weights for the linear, nonlinear, and constant blocks.

    Returns
    -------
    ops : dict with keys ``"c"`` (r,1), ``"A"`` (r,r) and, when nonlinear
    features are present, ``"N"`` (r,q).
    """
    Shat = np.asarray(Shat, dtype=float)
    dShat = np.asarray(dShat, dtype=float)
    r, k = Shat.shape
    r_out = dShat.shape[0]           # target row count (r, or m for a lifting map)
    blocks = [np.ones((1, k)), Shat]
    reg = [lam_const, *([lam_lin] * r)]
    has_nl = P_nl is not None and np.asarray(P_nl).size > 0
    if has_nl:
        P_nl = np.asarray(P_nl, dtype=float)
        q = P_nl.shape[0]
        blocks.append(P_nl)
        reg += [lam_nl] * q
    D = np.vstack(blocks).T  # (k, d)
    d = D.shape[1]
    Reg = np.diag(np.sqrt(np.asarray(reg)))
    # augmented Tikhonov least squares  
    # min ||D O^T - dShat^T||^2 + ||Reg O^T||^2
    D_aug = np.vstack([D, Reg])                   # (k + d, d)
    R_aug = np.vstack([dShat.T, np.zeros((d, r_out))])
    O_T, *_ = np.linalg.lstsq(D_aug, R_aug, rcond=None)
    O = O_T.T                                     # (r, d)
    ops = {"c": O[:, 0:1], "A": O[:, 1:1 + r]}
    if has_nl:
        ops["N"] = O[:, 1 + r:]
    return ops


def relative_fit_residual(Shat, dShat, P_nl, ops) -> float:
    """Relative training residual of the inferred ROM RHS (a diagnostic)."""
    Shat = np.asarray(Shat); dShat = np.asarray(dShat)
    pred = ops["c"] + ops["A"] @ Shat
    if "N" in ops and P_nl is not None and np.asarray(P_nl).size > 0:
        pred = pred + ops["N"] @ np.asarray(P_nl)
    return float(np.linalg.norm(dShat - pred) / (np.linalg.norm(dShat) + 1e-300))


# ===========================================================================
# ROM right-hand side and time integration (JAX)
# ===========================================================================


def _make_rhs(ops: Dict[str, jnp.ndarray], feature_fn: FeatureFn | None):
    c = ops["c"].reshape(-1)
    A = ops["A"]
    if "N" in ops and feature_fn is not None:
        N = ops["N"]

        def rhs(s):
            return c + A @ s + N @ feature_fn(s)
    else:
        def rhs(s):
            return c + A @ s
    return rhs


@partial(jax.jit, static_argnames=("rhs", "n_steps", "method"))
def _rollout(rhs, s0, dt, n_steps, method):
    """Integrate ``s' = rhs(s)`` for ``n_steps`` steps from ``s0``.

    ``method`` is ``"rk4"`` or ``"explicit"``.  Semi-implicit / CNAB steppers
    that need a factorized linear operator are handled in :func:`simulate`.
    """
    def rk4_step(s, _):
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * dt * k1)
        k3 = rhs(s + 0.5 * dt * k2)
        k4 = rhs(s + dt * k3)
        s_next = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return s_next, s_next

    def euler_step(s, _):
        s_next = s + dt * rhs(s)
        return s_next, s_next

    step = rk4_step if method == "rk4" else euler_step
    _, traj = jax.lax.scan(step, s0, None, length=n_steps)
    return traj  # (n_steps, r)


def _rollout_semi_implicit(ops, feature_fn, s0, dt, n_steps, *, theta=1.0):
    """Linearly-implicit rollout: (I - theta dt A) s+ = s + dt (c + N p(s)).

    ``theta = 1`` is backward-Euler on the linear part (L-stable, dissipative);
    ``theta = 0.5`` is the Crank-Nicolson variant used by the CNAB stepper.
    The nonlinear term is treated explicitly.  The linear factor is constant
    along a rollout, so it is factorized once.
    """
    r = s0.shape[0]
    A = ops["A"]
    c = ops["c"].reshape(-1)
    lhs = jnp.eye(r) - theta * dt * A
    lu, piv = jax.scipy.linalg.lu_factor(lhs)
    rhs_lin = (1.0 - theta) * dt * A
    has_nl = "N" in ops and feature_fn is not None
    N = ops["N"] if has_nl else None

    def step(s, _):
        expl = s + rhs_lin @ s + dt * c
        if has_nl:
            expl = expl + dt * (N @ feature_fn(s))
        s_next = jax.scipy.linalg.lu_solve((lu, piv), expl)
        return s_next, s_next

    _, traj = jax.lax.scan(step, s0, None, length=n_steps)
    return traj


def _gmg_rollout(a_ops, b_ops, feat_jac_fn, G_lift, s0, dt, n_steps):
    """Geometric manifold-Galerkin (RPG) rollout with RK4.

    Integrates ``shat' = M(shat)^{-1} (a(shat) + J_p^T b(shat))`` with the mass
    matrix ``M = I + J_p^T G_lift J_p`` (``G_lift = Vbar^T Vbar``).  ``M`` is SPD
    with ``||M - I|| <= L^2 - 1``, so the linear solve is well conditioned
    whenever the manifold's Lipschitz budget is finite.
    """
    r = s0.shape[0]
    I = jnp.eye(r)
    ca, Aa, Na = a_ops["c"].reshape(-1), a_ops["A"], a_ops["N"]
    cb, Ab, Nb = b_ops["c"].reshape(-1), b_ops["A"], b_ops["N"]

    def rhs(s):
        p, Jp = feat_jac_fn(s)   # (q,), (q, r)
        a = ca + Aa @ s + Na @ p
        b = cb + Ab @ s + Nb @ p
        M = I + Jp.T @ (G_lift @ Jp)
        return jnp.linalg.solve(M, a + Jp.T @ b)

    def rk4_step(s, _):
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * dt * k1)
        k3 = rhs(s + 0.5 * dt * k2)
        k4 = rhs(s + dt * k3)
        s_next = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return s_next, s_next

    _, traj = jax.lax.scan(rk4_step, s0, None, length=n_steps)
    return traj


def simulate_gmg(a_ops: Dict[str, np.ndarray], b_ops: Dict[str, np.ndarray],
                 feat_jac_fn, G_lift: np.ndarray, s0: np.ndarray,
                 n_steps: int, dt: float) -> np.ndarray:
    """Simulate the geometric (RPG) manifold ROM; return latent ``(r, n_steps+1)``.

    ``a_ops`` model ``V^T f`` and ``b_ops`` model ``Vbar^T f`` (both learned by
    Operator Inference); ``feat_jac_fn(s) -> (p, J_p)`` supplies the shared
    features and their Jacobian; ``G_lift = Vbar^T Vbar``.
    """
    a_j = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in a_ops.items()}
    b_j = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in b_ops.items()}
    G_j = jnp.asarray(np.asarray(G_lift, dtype=float))
    s0_j = jnp.asarray(np.asarray(s0, dtype=float).reshape(-1))
    roll = jax.jit(_gmg_rollout, static_argnames=("feat_jac_fn", "n_steps"))
    traj = np.asarray(roll(a_j, b_j, feat_jac_fn, G_j, s0_j,
                           jnp.asarray(float(dt)), int(n_steps)))
    full = np.empty((traj.shape[1], n_steps + 1))
    full[:, 0] = np.asarray(s0_j)
    full[:, 1:] = traj.T
    return full


def simulate(ops: Dict[str, np.ndarray], feature_fn: FeatureFn | None,
             s0: np.ndarray, n_steps: int, dt: float,
             *, 
             method: str = "rk4") -> np.ndarray:
    """Simulate the ROM and return the latent trajectory ``(r, n_steps+1)``.

    ``method`` is one of ``"rk4"``, ``"explicit"``, ``"semi_implicit"``
    (backward-Euler linear part), or ``"cnab"`` (Crank-Nicolson linear part,
    ``theta = 1/2``; the nonlinear term stays explicit Euler here, which is
    adequate for the mildly nonlinear advection-diffusion ROM).
    """
    ops_j = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in ops.items()}
    s0_j = jnp.asarray(np.asarray(s0, dtype=float).reshape(-1))
    dt_j = jnp.asarray(float(dt))
    if method in ("rk4", "explicit"):
        rhs = _make_rhs(ops_j, feature_fn)
        traj = _rollout(rhs, s0_j, dt_j, int(n_steps), method)
    elif method in ("semi_implicit", "cnab"):
        theta = 1.0 if method == "semi_implicit" else 0.5
        traj = _rollout_semi_implicit(
            ops_j, feature_fn, s0_j, dt_j, int(n_steps), theta=theta)
    else:
        raise ValueError(f"Unknown integration method {method!r}.")
    traj = np.asarray(traj)                        # (n_steps, r)
    full = np.empty((traj.shape[1], n_steps + 1))
    full[:, 0] = np.asarray(s0_j)
    full[:, 1:] = traj.T
    return full



# ===========================================================================
# Feature-function builders (model-type aware)
# ===========================================================================


def opm_feature_fn(inner) -> FeatureFn:
    """Active orthogonal-polynomial features of a fitted :class:`GreedyOPM`."""
    from .polynomials.arnoldi import arnoldi_eval
    basis = inner.arnoldi_basis_
    active = jnp.asarray(np.asarray(inner.active_nl_global_))

    def feats(s):
        Psi = arnoldi_eval(s.reshape(-1, 1), basis)   # (M, 1)
        return Psi[active, 0]
    return feats


def opm_feature_jac_fn(inner):
    """Active OPM features and their Jacobian ``(p, J_p)`` at a single point.

    ``J_p`` has shape ``(q, r)`` with ``J_p[k, j] = d psi_k / d shat_j``; both
    come from the differentiated Arnoldi recurrence, so the ROM's manifold
    metric reuses the exact gradients the certificate is built on.
    """
    from .polynomials.arnoldi import arnoldi_eval_grad
    basis = inner.arnoldi_basis_
    active = jnp.asarray(np.asarray(inner.active_nl_global_))

    def feats_jac(s):
        W, G = arnoldi_eval_grad(s.reshape(-1, 1), basis)   # (M,1), (M,r,1)
        return W[active, 0], G[active, :, 0]
    return feats_jac


# ===========================================================================
# Nonlinear feature values on a coordinate matrix (for the regression)
# ===========================================================================


def eval_features_matrix(feature_fn: FeatureFn | None,
                         Shat: np.ndarray) -> np.ndarray | None:
    """Evaluate ``feature_fn`` columnwise on ``Shat`` (r, k) -> (q, k)."""
    if feature_fn is None:
        return None
    Shat_j = jnp.asarray(np.asarray(Shat, dtype=float))
    P = jax.vmap(feature_fn, in_axes=1, out_axes=1)(Shat_j)
    return np.asarray(P)


# ===========================================================================
# Errors
# ===========================================================================


def relative_state_error(S_true: np.ndarray, S_approx: np.ndarray) -> float:
    """Relative Frobenius trajectory error ``||S_true - S_approx|| / ||S_true||``.

    Returns ``inf`` if the reconstruction contains non-finite values (a diverged
    ROM), matching the Julia ``compute_trajectory_error`` convention.
    """
    S_approx = np.asarray(S_approx)
    if not np.all(np.isfinite(S_approx)):
        return float("inf")
    S_true = np.asarray(S_true)
    return float(np.linalg.norm(S_true - S_approx) / np.linalg.norm(S_true))


# ===========================================================================
# Entrywise spline interpolation / affine extrapolation of operator sets
# ===========================================================================

_SUPPORTED_ORDERS = (3, 4, 5)


# ---------------------------------------------------------------------------
# Cubic natural spline (second-derivative form; matches the Julia reference)
# ---------------------------------------------------------------------------


def cubic_spline_second_derivs(z: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Natural cubic-spline second derivatives ``M`` at the knots."""
    z = np.asarray(z, dtype=float)
    f = np.asarray(f, dtype=float)
    n = z.shape[0] - 1
    h = np.diff(z)
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    A[0, 0] = 1.0
    A[n, n] = 1.0
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 6.0 * ((f[i + 1] - f[i]) / h[i] - (f[i] - f[i - 1]) / h[i - 1])
    return np.linalg.solve(A, b)


def cubic_spline_eval(z: np.ndarray, f: np.ndarray, M: np.ndarray,
                      zq: float) -> float:
    """Evaluate the natural cubic spline at an interior query ``zq``."""
    z = np.asarray(z, dtype=float)
    i = int(np.searchsorted(z, zq, side="right") - 1)
    i = min(max(i, 0), z.shape[0] - 2)
    h = z[i + 1] - z[i]
    t1 = M[i] * (z[i + 1] - zq) ** 3 / (6.0 * h)
    t2 = M[i + 1] * (zq - z[i]) ** 3 / (6.0 * h)
    t3 = (f[i] - M[i] * h * h / 6.0) * (z[i + 1] - zq) / h
    t4 = (f[i + 1] - M[i + 1] * h * h / 6.0) * (zq - z[i]) / h
    return float(t1 + t2 + t3 + t4)


# ---------------------------------------------------------------------------
# Quartic / quintic splines (full piecewise-coefficient systems)
# ---------------------------------------------------------------------------


def quartic_spline_coefficients(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Quartic-spline coefficients (5 per interval) with natural-type ends."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0] - 1
    h = np.diff(x)
    A = np.zeros((5 * n, 5 * n))
    b = np.zeros(5 * n)
    eq = 0
    for i in range(n):
        idx = 5 * i
        A[eq, idx] = 1.0
        b[eq] = y[i]
        eq += 1
        A[eq, idx:idx + 5] = [1.0, h[i], h[i] ** 2, h[i] ** 3, h[i] ** 4]
        b[eq] = y[i + 1]
        eq += 1
    for i in range(1, n):
        ip = 5 * (i - 1)
        ic = 5 * i
        hp = h[i - 1]
        A[eq, ip + 1:ip + 5] = [1.0, 2 * hp, 3 * hp ** 2, 4 * hp ** 3]
        A[eq, ic + 1] = -1.0
        eq += 1
    for i in range(1, n):
        ip = 5 * (i - 1)
        ic = 5 * i
        hp = h[i - 1]
        A[eq, ip + 2:ip + 5] = [2.0, 6 * hp, 12 * hp ** 2]
        A[eq, ic + 2] = -2.0
        eq += 1
    for i in range(1, n):
        ip = 5 * (i - 1)
        ic = 5 * i
        hp = h[i - 1]
        A[eq, ip + 3:ip + 5] = [6.0, 24 * hp]
        A[eq, ic + 3] = -6.0
        eq += 1
    A[eq, 2] = 2.0        # second derivative at first knot
    eq += 1
    A[eq, 3] = 6.0        # third derivative at first knot
    eq += 1
    il = 5 * (n - 1)
    hl = h[-1]
    A[eq, il + 2:il + 5] = [2.0, 6 * hl, 12 * hl ** 2]
    eq += 1
    if eq != 5 * n:
        raise RuntimeError("quartic spline: equation/unknown mismatch.")
    return np.linalg.solve(A, b)


def quintic_spline_coefficients(z: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Quintic-spline coefficients (6 per interval) with natural-type ends."""
    z = np.asarray(z, dtype=float)
    f = np.asarray(f, dtype=float)
    n = z.shape[0] - 1
    h = np.diff(z)
    A = np.zeros((6 * n, 6 * n))
    b = np.zeros(6 * n)
    for i in range(n):
        idx = 6 * i
        A[2 * i, idx] = 1.0
        A[2 * i + 1, idx:idx + 6] = [
            1.0, h[i], h[i] ** 2, h[i] ** 3, h[i] ** 4, h[i] ** 5]
        b[2 * i] = f[i]
        b[2 * i + 1] = f[i + 1]
    for i in range(1, n):
        row = 2 * n + 4 * (i - 1)
        hm = h[i - 1]
        ip = 6 * (i - 1)
        ic = 6 * i
        A[row, ip:ip + 6] = [0.0, 1.0, 2 * hm, 3 * hm ** 2,
                             4 * hm ** 3, 5 * hm ** 4]
        A[row, ic + 1] = -1.0
        A[row + 1, ip:ip + 6] = [0.0, 0.0, 2.0, 6 * hm,
                                 12 * hm ** 2, 20 * hm ** 3]
        A[row + 1, ic + 2] = -2.0
        A[row + 2, ip:ip + 6] = [0.0, 0.0, 0.0, 6.0, 24 * hm, 60 * hm ** 2]
        A[row + 2, ic + 3] = -6.0
        A[row + 3, ip:ip + 6] = [0.0, 0.0, 0.0, 0.0, 24.0, 120 * hm]
        A[row + 3, ic + 4] = -24.0
    A[6 * n - 4, 1] = 1.0          # first derivative at z[0]
    A[6 * n - 3, 2] = 2.0          # second derivative at z[0]
    hn = h[-1]
    il = 6 * (n - 1)
    A[6 * n - 2, il + 1:il + 6] = [1.0, 2 * hn, 3 * hn ** 2,
                                   4 * hn ** 3, 5 * hn ** 4]
    A[6 * n - 1, il + 2:il + 6] = [2.0, 6 * hn, 12 * hn ** 2, 20 * hn ** 3]
    return np.linalg.solve(A, b)


def _poly_spline_eval(z: np.ndarray, coeffs: np.ndarray, zq: float,
                      stride: int) -> float:
    """Evaluate a piecewise-polynomial spline (quartic/quintic) at ``zq``."""
    z = np.asarray(z, dtype=float)
    i = int(np.searchsorted(z, zq, side="right") - 1)
    i = min(max(i, 0), z.shape[0] - 2)
    hh = zq - z[i]
    c = coeffs[stride * i: stride * i + stride]
    return float(sum(c[p] * hh ** p for p in range(stride)))


# ---------------------------------------------------------------------------
# Scalar elementwise interpolation (interior) of a stack of matrices
# ---------------------------------------------------------------------------


def interpolate_matrix_elements(z: Sequence[float], matrices: Sequence[np.ndarray],
                                zq: float, order: int = 3) -> np.ndarray:
    """Interpolate every entry of a set of matrices at ``zq`` (interior query).

    ``matrices[k]`` is the matrix at parameter ``z[k]``; all share one shape.
    Mirrors the Julia ``interpolate_matrix_elements``.

    Arguments
    ---------
    z : (P,) strictly increasing parameter nodes.
    matrices : list of (a, b) real matrices, one per node.
    zq : query parameter, expected within ``[z[0], z[-1]]``.
    order : 3 (cubic natural), 4 (quartic), or 5 (quintic).
    """
    if order not in _SUPPORTED_ORDERS:
        raise ValueError(f"order must be one of {_SUPPORTED_ORDERS}; got {order}.")
    z = np.asarray(z, dtype=float)
    stack = np.stack([np.asarray(m, dtype=float) for m in matrices], axis=0)
    P, a, b = stack.shape
    out = np.empty((a, b))
    for i in range(a):
        for j in range(b):
            fvals = stack[:, i, j]
            if order == 3:
                M = cubic_spline_second_derivs(z, fvals)
                out[i, j] = cubic_spline_eval(z, fvals, M, zq)
            elif order == 4:
                c = quartic_spline_coefficients(z, fvals)
                out[i, j] = _poly_spline_eval(z, c, zq, 5)
            else:
                c = quintic_spline_coefficients(z, fvals)
                out[i, j] = _poly_spline_eval(z, c, zq, 6)
    return out


# ---------------------------------------------------------------------------
# Affine (degree-1) elementwise extrapolation
# ---------------------------------------------------------------------------


def affine_extrapolate_matrix_elements(z: Sequence[float],
                                       matrices: Sequence[np.ndarray],
                                       zq: float) -> np.ndarray:
    """Per-entry least-squares affine fit ``a*z + b`` evaluated at ``zq``.

    Uses all nodes, so it is robust and exact whenever the entries are affine
    in the parameter (the linear-PDE case).  Suitable for queries outside the
    knot range where a high-order spline would diverge.
    """
    z = np.asarray(z, dtype=float)
    stack = np.stack([np.asarray(m, dtype=float) for m in matrices], axis=0)
    P, a, b = stack.shape
    # least-squares slope/intercept per entry in one solve
    zbar = z.mean()
    zc = z - zbar
    denom = float(np.dot(zc, zc))
    F = stack.reshape(P, a * b)
    fbar = F.mean(axis=0)
    slope = (zc[:, None] * (F - fbar[None, :])).sum(axis=0) / denom
    val = fbar + slope * (zq - zbar)
    return val.reshape(a, b)


# ---------------------------------------------------------------------------
# Operator-dictionary convenience wrappers
# ---------------------------------------------------------------------------


def interp_operator_dict(z: Sequence[float],
                         op_dicts: Sequence[Dict[str, np.ndarray]],
                         zq: float, *, order: int = 3,
                         keys: Sequence[str] | None = None
                         ) -> Dict[str, np.ndarray]:
    """Interpolate a per-parameter operator dictionary at an interior ``zq``.

    ``op_dicts[k]`` maps an operator name (e.g. ``"c"``, ``"A"``, ``"N"``) to
    the operator at parameter ``z[k]``.  Returns the interpolated dictionary.
    """
    keys = list(op_dicts[0].keys()) if keys is None else list(keys)
    return {
        name: interpolate_matrix_elements(
            z, [od[name] for od in op_dicts], zq, order=order)
        for name in keys
    }


def extrap_operator_dict(z: Sequence[float],
                         op_dicts: Sequence[Dict[str, np.ndarray]],
                         zq: float, *,
                         keys: Sequence[str] | None = None
                         ) -> Dict[str, np.ndarray]:
    """Affine-extrapolate a per-parameter operator dictionary at ``zq``."""
    keys = list(op_dicts[0].keys()) if keys is None else list(keys)
    return {
        name: affine_extrapolate_matrix_elements(
            z, [od[name] for od in op_dicts], zq)
        for name in keys
    }


def nearest_operator_dict(z: Sequence[float],
                          op_dicts: Sequence[Dict[str, np.ndarray]],
                          zq: float, *,
                          keys: Sequence[str] | None = None
                          ) -> Dict[str, np.ndarray]:
    """Nearest-parameter operator dictionary (naive extrapolation fallback)."""
    z = np.asarray(z, dtype=float)
    k = int(np.argmin(np.abs(z - zq)))
    keys = list(op_dicts[0].keys()) if keys is None else list(keys)
    return {name: np.asarray(op_dicts[k][name], dtype=float) for name in keys}


#: Backwards-compatible alias (the note calls the geometric model RPG in places).
simulate_rpg = simulate_gmg


# ===========================================================================
# Affine-in-parameter Operator Inference
# ===========================================================================


def infer_operators_affine(
    Shat_list: Sequence[np.ndarray],
    dShat_list: Sequence[np.ndarray],
    P_list: Sequence[np.ndarray | None],
    taus: Sequence[float],
    *,
    lam_lin: float = 0.0,
    lam_nl: float = 0.0,
    lam_const: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Pooled affine-in-parameter OpInf across the training parameters.

    For a FOM whose operator depends affinely on the parameter (e.g. linear
    advection-diffusion, ``A(c) = -c D1 + nu D2``), every projected reduced
    operator inherits the affine structure exactly, so the reduced RHS is

        dShat = (O0 + tau O1) d(shat),      d = [1; shat; p(shat)],

    with ``tau`` the centered/scaled parameter.  This routine fits ``(O0, O1)``
    in ONE regression pooling all training-parameter trajectories -- the
    per-snapshot design row is ``[d; tau_k d]`` -- replacing the per-parameter
    fits + entrywise operator interpolation (more data per unknown, no
    interpolation error, and operators that vary linearly in the parameter).

    Arguments
    ---------
    Shat_list, dShat_list, P_list : per-parameter reduced coordinates,
        reduced velocities, and nonlinear feature values (``P`` may be None).
    taus : the centered/scaled parameter value of each trajectory.
    lam_* : order-wise Tikhonov weights, applied identically to both the
        ``O0`` and ``O1`` blocks.

    Returns
    -------
    (ops0, ops1) : operator dicts with keys ``"c"``, ``"A"`` and (with
    nonlinear features) ``"N"``; the operator at parameter ``tau`` is
    ``{key: ops0[key] + tau * ops1[key]}`` (see :func:`affine_ops_at`).
    """
    r = int(np.asarray(Shat_list[0]).shape[0])
    r_out = int(np.asarray(dShat_list[0]).shape[0])
    has_nl = P_list[0] is not None and np.asarray(P_list[0]).size > 0
    rows_D, rows_R = [], []
    q = 0
    for Shat, dShat, P_nl, tau in zip(Shat_list, dShat_list, P_list, taus):
        Shat = np.asarray(Shat, dtype=float)
        k = Shat.shape[1]
        blocks = [np.ones((1, k)), Shat]
        if has_nl:
            P_nl = np.asarray(P_nl, dtype=float)
            q = P_nl.shape[0]
            blocks.append(P_nl)
        Dk = np.vstack(blocks).T                       # (k, d)
        rows_D.append(np.hstack([Dk, float(tau) * Dk]))
        rows_R.append(np.asarray(dShat, dtype=float).T)
    D = np.vstack(rows_D)                              # (K, 2d)
    R = np.vstack(rows_R)                              # (K, r_out)
    reg1 = [lam_const, *([lam_lin] * r)] + ([lam_nl] * q if has_nl else [])
    reg = np.asarray(reg1 + reg1, dtype=float)
    d2 = D.shape[1]
    D_aug = np.vstack([D, np.diag(np.sqrt(reg))])
    R_aug = np.vstack([R, np.zeros((d2, r_out))])
    O_T, *_ = np.linalg.lstsq(D_aug, R_aug, rcond=None)
    O = O_T.T                                          # (r_out, 2d)
    d1 = d2 // 2

    def _split(Oh):
        ops = {"c": Oh[:, 0:1], "A": Oh[:, 1:1 + r]}
        if has_nl:
            ops["N"] = Oh[:, 1 + r:]
        return ops

    return _split(O[:, :d1]), _split(O[:, d1:])


def affine_ops_at(ops0: Dict[str, np.ndarray], ops1: Dict[str, np.ndarray],
                  tau: float) -> Dict[str, np.ndarray]:
    """Evaluate the affine operator family at the scaled parameter ``tau``."""
    return {key: np.asarray(ops0[key]) + float(tau) * np.asarray(ops1[key])
            for key in ops0}
