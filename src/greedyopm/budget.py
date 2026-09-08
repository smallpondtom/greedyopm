"""Lipschitz budget calibration and the one-step lifting solution.

Closed-form machinery of the Lipschitz budget Lagrangian

    min_Vbar  1/2 ||P^T Vbar^T - E^T||_F^2  +  lambda * sum_k eta_k ||vbar_k||_2 ,

posed on empirically orthonormal features ((1/n) P P^T = I).  Because the
Gram matrix is diagonal, the problem separates over columns and a single
proximal-gradient step is exact (Theorem `onestep`):

    vbar_k*(lambda) = (1 - lambda eta_k / (n ||r_k||))_+ r_k ,
    r_k = (1/n) E [P]_{k,:}^T .

The stability budget B(Vbar) = sum_k eta_k ||vbar_k|| certifies the decoder
Lipschitz constant through  L_hat = sqrt(1 + B^2), so the user's budget
L > 1 is the convex constraint B <= B_L = sqrt(L^2 - 1).  Evaluated along
the one-step path, the attained budget

    B(lambda) = sum_k eta_k (||r_k|| - lambda eta_k / n)_+

is piecewise linear and strictly decreasing where positive, so the optimal
multiplier lambda* with B(lambda*) = B_L is found exactly by one sort of the
breakpoints rho_k = n ||r_k|| / eta_k and two prefix sums (Theorem
`budget_lambda`, Algorithm `Calibrate`) -- no iterative dual ascent.

All functions are pure and jit/vmap-compatible; ``n`` must be passed as a
float (weak) scalar so shapes stay static.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


#: Floor used in divisions by eta_k; features with eta_k = 0 carry no
#: stability risk and must stay active with zero shrinkage.
_ETA_FLOOR = 1e-150


@jax.jit
def calibrate_multiplier(
    r_norms: jnp.ndarray,
    eta: jnp.ndarray,
    B_L: jnp.ndarray,
    n: float,
) -> jnp.ndarray:
    """Exact Lipschitz-budget calibration (Algorithm `Calibrate`).

    Returns the smallest multiplier lambda* >= 0 such that the attained
    budget satisfies B(lambda*) <= B_L, attained with equality whenever the
    constraint is active: lambda* = 0 if B(0) <= B_L, otherwise the unique
    root of the piecewise-linear B(lambda) = B_L, located by sorting the
    breakpoints rho_k = n ||r_k|| / eta_k and scanning the prefix sums
    a_j = sum eta_(i) ||r_(i)||,  b_j = (1/n) sum eta_(i)^2.

    Arguments
    ---------
    r_norms : (m,)
        Residual-correlation norms ||r_k||_2.
    eta : (m,)
        Sup-gradient bounds eta_k >= 0 of the nonlinear features.
    B_L : scalar
        Budget bound sqrt(L^2 - 1).
    n : float
        Number of training snapshots.

    Returns
    -------
    lambda_star : scalar (same dtype as ``r_norms``)
    """
    w = eta * r_norms                                   # eta_k ||r_k||
    B0 = jnp.sum(w)

    eta_safe = jnp.maximum(eta, _ETA_FLOOR)
    rho = n * r_norms / eta_safe                        # breakpoints
    order = jnp.argsort(-rho)                           # rho_(1) >= rho_(2) >= ...
    a = jnp.cumsum(w[order])                            # prefix sums a_j
    b = jnp.cumsum(eta_safe[order] ** 2) / n            # prefix sums b_j
    rho_sorted = rho[order]

    # B(rho_(j)) = a_j - rho_(j) b_j is nondecreasing in j; j* is the last
    # feasible index (j = 1 is always feasible since B(rho_(1)) = 0).
    feasible = (a - rho_sorted * b) <= B_L
    idx = jnp.arange(r_norms.shape[0])
    j_star = jnp.max(jnp.where(feasible, idx, -1))
    j_star = jnp.maximum(j_star, 0)

    lam = (a[j_star] - B_L) / b[j_star]
    lam = jnp.maximum(lam, 0.0)
    return jnp.where(B0 <= B_L, jnp.zeros_like(lam), lam)


@jax.jit
def shrink_factors(
    r_norms: jnp.ndarray,
    eta: jnp.ndarray,
    lam: jnp.ndarray,
    n: float,
) -> jnp.ndarray:
    """Columnwise group soft-threshold factors of the one-step solution.

    ``vbar_k*(lambda) = shrink_k * r_k`` with
    ``shrink_k = (1 - lambda eta_k / (n ||r_k||))_+`` (Theorem `onestep`);
    a feature is active iff its factor is positive.

    Arguments
    ---------
    r_norms : (m,)
    eta : (m,)
    lam : scalar
    n : float

    Returns
    -------
    shrink : (m,)  in [0, 1].
    """
    denom = jnp.maximum(n * r_norms, _ETA_FLOOR)
    return jnp.maximum(1.0 - lam * eta / denom, 0.0)


@jax.jit
def feature_gains(
    r_norms: jnp.ndarray,
    eta: jnp.ndarray,
    lam: jnp.ndarray,
    n: float,
) -> jnp.ndarray:
    """Per-feature training-error reductions at multiplier ``lam``.

    ``G_k(lambda) = (n^2 ||r_k||^2 - lambda^2 eta_k^2)_+ / n`` (Proposition
    `accgain`): the squared training error of the solved manifold is
    ``||E||_F^2 - sum_k G_k(lambda)``, and G_k vanishes exactly on the
    inactive set.

    Arguments
    ---------
    r_norms : (m,)
    eta : (m,)
    lam : scalar
    n : float

    Returns
    -------
    gains : (m,)  nonnegative.
    """
    return jnp.maximum((n * r_norms) ** 2 - (lam * eta) ** 2, 0.0) / n


@jax.jit
def attained_budget(
    r_norms: jnp.ndarray,
    eta: jnp.ndarray,
    lam: jnp.ndarray,
    n: float,
) -> jnp.ndarray:
    """Attained stability budget ``B(lambda)`` of the one-step solution.

    ``B(lambda) = sum_k eta_k (||r_k|| - lambda eta_k / n)_+`` (Lemma
    `budget_path`); equals ``sum_k eta_k ||vbar_k*(lambda)||``.
    """
    return jnp.sum(eta * jnp.maximum(r_norms - lam * eta / n, 0.0))


def certificate_from_budget(B: jnp.ndarray) -> jnp.ndarray:
    """Lipschitz certificate ``L_hat = sqrt(1 + B^2)`` from a budget value."""
    return jnp.sqrt(1.0 + B ** 2)
