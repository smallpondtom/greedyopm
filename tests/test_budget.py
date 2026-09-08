"""Verification of the Lipschitz-budget machinery against the paper.

Checks Theorem `onestep` (one-step solution vs an iterative FISTA
reference), Lemma `budget_path` / Theorem `budget_lambda` (exact
calibration: B(lambda*) = B_L when active), and Proposition `accgain`
(additive accuracy decomposition).
"""

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from greedyopm.budget import (  # noqa: E402
    attained_budget,
    calibrate_multiplier,
    feature_gains,
    shrink_factors,
)


def _random_problem(seed, d=40, m=25, n=60):
    """Random orthonormal-feature regression problem ((1/n) P P^T = I)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, m)))
    P = np.sqrt(n) * Q.T                       # (m, n), (1/n) P P^T = I
    E = rng.standard_normal((d, n))
    eta = np.abs(rng.standard_normal(m)) + 0.1
    R = (1.0 / n) * E @ P.T                    # residual correlations r_k
    return jnp.asarray(P), jnp.asarray(E), jnp.asarray(R), jnp.asarray(eta), n


def _fista_reference(P, E, eta, lam, n, n_iter=20000):
    """Iterative proximal-gradient reference for the budget Lagrangian."""
    P, E, eta = np.asarray(P), np.asarray(E), np.asarray(eta)
    d, m = E.shape[0], P.shape[0]
    G = P @ P.T
    EPt = E @ P.T
    Lip = np.linalg.eigvalsh(G)[-1]
    V = np.zeros((d, m))
    Y, t = V.copy(), 1.0
    for _ in range(n_iter):
        grad = Y @ G - EPt
        Z = Y - grad / Lip
        col = np.linalg.norm(Z, axis=0)
        scale = np.maximum(0.0, 1.0 - (lam * eta / Lip) / np.maximum(col, 1e-300))
        V_new = Z * scale[None, :]
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Y = V_new + ((t - 1.0) / t_new) * (V_new - V)
        V, t = V_new, t_new
    return V


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_onestep_solution_matches_fista(seed):
    """The closed-form soft-threshold is the global Lagrangian minimizer."""
    P, E, R, eta, n = _random_problem(seed)
    r_norms = jnp.linalg.norm(R, axis=0)
    lam = 0.3 * float(n * jnp.max(r_norms) / jnp.min(eta))

    shrink = shrink_factors(r_norms, eta, jnp.asarray(lam), float(n))
    V_closed = np.asarray(R) * np.asarray(shrink)[None, :]
    V_ref = _fista_reference(P, E, eta, lam, n)
    assert np.allclose(V_closed, V_ref, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_calibration_hits_budget_exactly(seed):
    """Active constraint: B(lambda*) = B_L; inactive: lambda* = 0."""
    _, _, R, eta, n = _random_problem(seed)
    r_norms = jnp.linalg.norm(R, axis=0)
    B0 = float(jnp.sum(eta * r_norms))

    # Active regime: budget below B(0).
    for frac in (0.9, 0.5, 0.1, 1e-3):
        B_L = jnp.asarray(frac * B0)
        lam = calibrate_multiplier(r_norms, eta, B_L, float(n))
        B_att = float(attained_budget(r_norms, eta, lam, float(n)))
        assert lam > 0.0
        assert np.isclose(B_att, float(B_L), rtol=1e-10)

    # Inactive regime: budget above B(0).
    lam = calibrate_multiplier(r_norms, eta, jnp.asarray(2.0 * B0), float(n))
    assert float(lam) == 0.0


def test_calibration_zero_eta_features_stay_free():
    """Features with eta_k = 0 carry no risk: active, unshrunk, free."""
    n = 50
    r_norms = jnp.asarray([1.0, 2.0, 3.0])
    eta = jnp.asarray([0.0, 1.0, 1.0])
    B_L = jnp.asarray(1.0)                     # < B(0) = 5
    lam = calibrate_multiplier(r_norms, eta, B_L, float(n))
    shrink = shrink_factors(r_norms, eta, lam, float(n))
    assert float(shrink[0]) == 1.0             # zero-risk feature untouched
    B_att = float(attained_budget(r_norms, eta, lam, float(n)))
    assert np.isclose(B_att, 1.0, rtol=1e-12)


@pytest.mark.parametrize("seed", [0, 1])
def test_additive_accuracy_decomposition(seed):
    """||P^T Vbar*^T - E^T||_F^2 = ||E||_F^2 - sum_k G_k(lambda)."""
    P, E, R, eta, n = _random_problem(seed)
    r_norms = jnp.linalg.norm(R, axis=0)
    lam = jnp.asarray(0.2 * float(n * jnp.max(r_norms) / jnp.min(eta)))

    shrink = shrink_factors(r_norms, eta, lam, float(n))
    Vbar = R * shrink[None, :]
    misfit = float(jnp.linalg.norm(Vbar @ P - E, "fro") ** 2)
    gains = feature_gains(r_norms, eta, lam, float(n))
    predicted = float(jnp.linalg.norm(E, "fro") ** 2 - jnp.sum(gains))
    assert np.isclose(misfit, predicted, rtol=1e-10)


def test_budget_path_piecewise_linear_monotone():
    """B(lambda) is nonincreasing and hits 0 at the largest breakpoint."""
    _, _, R, eta, n = _random_problem(7)
    r_norms = jnp.linalg.norm(R, axis=0)
    rho_max = float(jnp.max(n * r_norms / eta))
    lams = np.linspace(0.0, 1.1 * rho_max, 200)
    B = [float(attained_budget(r_norms, eta, jnp.asarray(l), float(n)))
         for l in lams]
    assert all(b1 >= b2 - 1e-12 for b1, b2 in zip(B, B[1:]))
    assert np.isclose(B[-1], 0.0, atol=1e-12)
