"""Tests for the budget-constrained selection and Lipschitz certificate.

Covers:
* budget=None backward compatibility (unconstrained selection),
* budget enforcement by construction (certificate <= L after fit),
* monotonicity of the admitted set / certificate in the budget L,
* certificate validity (pointwise Jacobian norms <= certificate on the
  dilated probe cloud),
* the cheap (gamma_sp, budget) re-selection path (set_sparsity).
"""

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from greedyopm import GreedyOPM  # noqa: E402


@pytest.fixture(scope="module")
def data():
    x = np.linspace(0.0, 1.0, 160, endpoint=False)
    t = np.linspace(0.0, 1.0, 220)
    S = np.exp(-((x[:, None] - (0.2 + 0.5 * t[None, :])) ** 2) / 2e-3)
    return jnp.asarray(S)


class TestBudgetedSelection:
    def test_unconstrained_matches_legacy(self, data):
        """budget=None keeps every eligible feature active."""
        m = GreedyOPM(r=4, s=12, p=2, gamma_sp=0.0).fit(data)
        assert m.n_selected_nonlinear == m.m_nl_

    def test_budget_requires_arnoldi(self):
        with pytest.raises(ValueError):
            GreedyOPM(r=2, s=6, p=2, budget=10.0, ortho_method="qr")

    def test_budget_must_exceed_one(self):
        with pytest.raises(ValueError):
            GreedyOPM(r=2, s=6, p=2, budget=0.5)

    def test_certificate_below_budget_by_construction(self, data):
        for L in (1e5, 1e2, 10.0, 2.0):
            m = GreedyOPM(
                r=4, s=12, p=3, budget=L, grad_dilation=0.2,
            ).fit(data)
            assert m.certificate_ <= L * (1.0 + 1e-9)

    def test_monotone_in_budget(self, data):
        """Looser budgets admit at least as much gain (same fitted basis)."""
        m = GreedyOPM(r=4, s=12, p=3, grad_dilation=0.1).fit(data)
        errs, acts = [], []
        for L in (2.0, 10.0, 1e3, np.inf):
            m.set_sparsity(0.0, budget=L)
            errs.append(m.reconstruction_error(data))
            acts.append(m.n_selected_nonlinear)
        assert all(a <= b for a, b in zip(acts, acts[1:]))
        assert all(e >= f - 1e-12 for e, f in zip(errs, errs[1:]))

    def test_certificate_validity(self, data):
        """Pointwise Jacobian norms never exceed the certificate."""
        m = GreedyOPM(
            r=4, s=12, p=3, budget=1e3, grad_dilation=0.25,
            n_grad_samples=40,
        ).fit(data)
        probe = (1.0 + 0.25) * m.S_hat_[:, ::5]
        pointwise = m.decoder_jacobian_norms(probe)
        assert float(jnp.max(pointwise)) <= m.certificate_ * (1.0 + 1e-9)

    def test_admission_between_linear_and_full(self, data):
        """Budgeted error sits between the full and linear-only errors."""
        m = GreedyOPM(r=4, s=12, p=2, grad_dilation=0.0).fit(data)
        m.set_sparsity(0.0, budget=None)
        full_err = m.reconstruction_error(data)
        lin_err = m.linear_error(data)
        m.set_sparsity(0.0, budget=50.0)
        budget_err = m.reconstruction_error(data)
        assert full_err - 1e-12 <= budget_err <= lin_err + 1e-12

    def test_resweep_gamma_sp_keeps_budget(self, data):
        m = GreedyOPM(r=4, s=12, p=3, budget=30.0, grad_dilation=0.1).fit(data)
        for gsp in (0.0, 1e-3, 1e-1):
            m.set_sparsity(gsp)
            assert m.certificate_ <= 30.0 * (1.0 + 1e-9)
