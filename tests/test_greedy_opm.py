"""End-to-end verification of GreedyOPM against the paper.

Checks Fact `P_properties` (orthonormality / decorrelation of the feature
matrix), the enforcement identity of Theorem `budget_lambda`(iii)
(L_hat = min(sqrt(1 + B(0)^2), L)), the gradient recurrence against finite
differences, and basic manifold consistency (train error decreases with L,
unregularized limit equals exact least squares).
"""

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from greedyopm import GreedyFixedOPM, GreedyOPM  # noqa: E402
from greedyopm.polynomials.arnoldi import arnoldi_eval_grad  # noqa: E402


def _snapshots(d=60, n=120, seed=0):
    """Smooth low-rank-ish synthetic snapshot matrix (d, n)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    x = np.linspace(0.0, 1.0, d)[:, None]
    S = sum(
        np.exp(-((x - 0.2 - 0.6 * tk) ** 2) / 0.01)[:, 0][:, None] * 0.0
        for tk in [0]
    )
    S = np.zeros((d, n))
    for j, tj in enumerate(t):
        S[:, j] = np.exp(-((x[:, 0] - 0.2 - 0.6 * tj) ** 2) / 0.01)
    S += 0.01 * rng.standard_normal((d, n))
    return jnp.asarray(S)


@pytest.fixture(scope="module")
def fitted():
    S = _snapshots()
    model = GreedyOPM(r=4, s=12, p=[2, 3], L=10.0, delta=0.1).fit(S)
    return model, S


def test_feature_orthonormality_and_decorrelation(fitted):
    """Fact `P_properties`: (1/n) P P^T = I,  P S_hat^T = 0,  P 1 = 0."""
    model, _ = fitted
    P = model.Psi_[model.n_linear_:]
    n = P.shape[1]
    G = (1.0 / n) * (P @ P.T)
    assert float(jnp.max(jnp.abs(G - jnp.eye(G.shape[0])))) < 1e-8
    assert float(jnp.max(jnp.abs(P @ model.S_hat_.T))) / n < 1e-8
    assert float(jnp.max(jnp.abs(P @ jnp.ones((n, 1))))) / n < 1e-8


def test_certificate_enforcement(fitted):
    """L_hat = min(sqrt(1 + B(0)^2), L) <= L, and matches a-posteriori."""
    model, _ = fitted
    B0 = float(jnp.sum(model.eta_ * model.r_norms_))
    expected = min(np.sqrt(1.0 + B0 ** 2), model.L)
    assert model.certificate_ <= model.L + 1e-9
    assert np.isclose(model.certificate_, expected, rtol=1e-8)
    # a-posteriori recomputation on the fit-time probe set agrees
    assert np.isclose(
        model.lipschitz_certificate(), model.certificate_, rtol=1e-8,
    )


def test_certificate_bounds_jacobian(fitted):
    """Pointwise ||J_Gamma||_2 on the dilated cloud never exceeds L_hat."""
    model, _ = fitted
    S_hat_dilated = (1.0 + model.delta) * model.S_hat_
    norms = model.decoder_jacobian_norms(S_hat_dilated)
    assert float(jnp.max(norms)) <= model.certificate_ * (1.0 + 1e-8)


def test_unregularized_limit_is_least_squares(fitted):
    """L = inf: lambda* = 0, all features active, Vbar = (1/n) E P^T."""
    model, S = fitted
    model.set_budget(np.inf)
    assert model.lambda_ == 0.0
    assert model.n_active_nonlinear == model.m_nl_
    assert np.allclose(np.asarray(model.Vbar_nl_), np.asarray(model.C_nl_))
    model.set_budget(10.0)                      # restore


def test_train_error_monotone_in_budget(fitted):
    """A larger budget can only reduce the training reconstruction error."""
    model, S = fitted
    errs = []
    for L in (1.5, 3.0, 10.0, 100.0):
        model.set_budget(L)
        errs.append(model.reconstruction_error(S))
        assert model.certificate_ <= L + 1e-9
    assert all(e1 >= e2 - 1e-12 for e1, e2 in zip(errs, errs[1:]))
    model.set_budget(10.0)


def test_gradient_recurrence_vs_finite_differences(fitted):
    """The Arnoldi gradient recurrence matches central finite differences."""
    model, _ = fitted
    S0 = model.S_hat_[:, ::37]                  # a few probe points
    _, G = arnoldi_eval_grad(S0, model.arnoldi_basis_)
    h = 1e-6
    scale = float(jnp.max(jnp.abs(S0)))
    for j in range(S0.shape[0]):
        Sp = S0.at[j].add(h * scale)
        Sm = S0.at[j].add(-h * scale)
        from greedyopm.polynomials.arnoldi import arnoldi_eval
        fd = (arnoldi_eval(Sp, model.arnoldi_basis_)
              - arnoldi_eval(Sm, model.arnoldi_basis_)) / (2 * h * scale)
        err = float(jnp.max(jnp.abs(fd - G[:, j, :])))
        ref = float(jnp.max(jnp.abs(G))) + 1.0
        assert err / ref < 1e-6


def test_score_matches_realized_error_drop():
    """The committed candidate's score equals sigma^2 + realized poly gain.

    Sanity check of Proposition `mode_decomp` on the last iteration: the
    final model's calibrated gains, recomputed from the exact residual,
    should be close to the (pool-truncated) gains used in the greedy score.
    """
    S = _snapshots(seed=3)
    model = GreedyOPM(r=3, s=10, p=[2], L=50.0, delta=0.05).fit(S)
    from greedyopm.budget import feature_gains
    n = float(model.Psi_.shape[1])
    gains = feature_gains(
        model.r_norms_, model.eta_, jnp.asarray(model.lambda_), n,
    )
    rec = model.iteration_records_[-1]
    # pool truncation and residual update make these approximately equal
    assert np.isclose(
        rec["poly_gain"], float(jnp.sum(gains)), rtol=0.2,
    )


def test_fixed_opm_runs_and_interpolates_better_on_train():
    """GreedyFixedOPM fits and beats POD-truncation error on training data."""
    S = _snapshots(seed=1)
    model = GreedyFixedOPM(r=3, s=10, p=[2]).fit(S)
    S_hat = model.encode(S)
    lin = model.s_ref_ + model.U_r_ @ S_hat
    err_lin = float(jnp.linalg.norm(S - lin, "fro"))
    err_nl = float(jnp.linalg.norm(S - model.reconstruct(S), "fro"))
    assert err_nl < err_lin
