"""Tests for the stability-regularized greedy OPM (Jacobian control).

Covers:
* gradient recurrence vs. jax.jacfwd (exactness of Proposition `gradient
  recurrence`),
* mu = 0 backward compatibility (legacy scoring path, legacy shrinkage),
* monotonicity of the Lipschitz certificate in mu,
* bounded accuracy loss on training data (Lemma `bounded accuracy loss`),
* OOD smoke test on a traveling-wave toy problem.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from greedyopm import GreedyOPM
from greedyopm.polynomials.arnoldi import (
    arnoldi_build,
    arnoldi_eval,
    arnoldi_eval_grad,
)
from greedyopm.polynomials.multindex import (
    graded_lex_multi_indices_for_degrees,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Toy data: traveling Gaussian bump (transport-dominated, slow n-width decay)
# ---------------------------------------------------------------------------


def traveling_wave(n_t: int, t_max: float, d: int = 120) -> jnp.ndarray:
    x = np.linspace(-1.0, 1.0, d)
    t = np.linspace(0.0, t_max, n_t)
    S = np.exp(-((x[:, None] - (-0.6 + t[None, :])) ** 2) / (2 * 0.08**2))
    return jnp.asarray(S)


# ---------------------------------------------------------------------------
# Gradient recurrence
# ---------------------------------------------------------------------------


class TestGradientRecurrence:
    def _basis(self, r=2, n=40, p=3, seed=0):
        rng = np.random.default_rng(seed)
        S_hat = jnp.asarray(rng.standard_normal((r, n)))
        mi = np.asarray(
            graded_lex_multi_indices_for_degrees(r, list(range(2, p + 1))),
            dtype=np.int32,
        )
        _, basis = arnoldi_build(S_hat, mi, n_reorth=1)
        return S_hat, basis

    def test_values_match_eval(self):
        S_hat, basis = self._basis()
        W_ref = arnoldi_eval(S_hat, basis)
        W, _ = arnoldi_eval_grad(S_hat, basis)
        np.testing.assert_allclose(np.asarray(W), np.asarray(W_ref), atol=1e-12)

    @pytest.mark.parametrize("r,p", [(1, 4), (2, 3), (3, 2)])
    def test_grad_matches_jacfwd(self, r, p):
        rng = np.random.default_rng(1)
        n = 50
        S_hat = jnp.asarray(rng.standard_normal((r, n)))
        mi = np.asarray(
            graded_lex_multi_indices_for_degrees(r, list(range(2, p + 1))),
            dtype=np.int32,
        )
        _, basis = arnoldi_build(S_hat, mi, n_reorth=1)

        # a handful of probe points, in and out of distribution
        pts = jnp.asarray(rng.standard_normal((r, 7)) * 1.5)
        _, G = arnoldi_eval_grad(pts, basis)          # (M, r, 7)

        def features_at(x):                           # x: (r,)
            return arnoldi_eval(x[:, None], basis)[:, 0]

        J_ref = jax.vmap(jax.jacfwd(features_at), in_axes=1, out_axes=2)(pts)
        np.testing.assert_allclose(
            np.asarray(G), np.asarray(J_ref), rtol=1e-9, atol=1e-9,
        )

    def test_grad_of_constant_and_linear(self):
        """grad psi_0 = 0; grad of linear features is constant in s_hat."""
        S_hat, basis = self._basis(r=2, n=30, p=2)
        pts = jnp.asarray(np.random.default_rng(2).standard_normal((2, 5)))
        _, G = arnoldi_eval_grad(pts, basis)
        assert float(jnp.max(jnp.abs(G[0]))) == 0.0
        # linear rows: gradient identical across points
        for k in (1, 2):
            spread = jnp.max(jnp.abs(G[k] - G[k][:, :1]))
            assert float(spread) < 1e-10


# ---------------------------------------------------------------------------
# GreedyOPM with stability regularization
# ---------------------------------------------------------------------------


class TestStabilityRegularizedGreedy:
    @pytest.fixture(scope="class")
    def data(self):
        return traveling_wave(n_t=160, t_max=0.9)

    def test_mu_zero_is_legacy(self, data):
        """mu=0 must reproduce the unregularized model bit-for-bit."""
        kw = dict(gamma_sp=1e-3, gamma=1e-8, ortho_method="arnoldi")
        m0 = GreedyOPM(r=4, s=12, p=2, **kw).fit(data)
        m1 = GreedyOPM(r=4, s=12, p=2, mu=0.0, **kw).fit(data)
        assert list(m0.J_) == list(m1.J_)
        np.testing.assert_allclose(
            np.asarray(m0.Vbar_nl_), np.asarray(m1.Vbar_nl_), atol=0.0,
        )
        assert m0.reconstruction_error(data) == m1.reconstruction_error(data)

    def test_mu_requires_arnoldi(self):
        with pytest.raises(ValueError):
            GreedyOPM(r=2, s=6, p=2, mu=1.0, ortho_method="qr")

    def test_certificate_decreases_with_mu(self, data):
        certs = []
        for mu in (0.0, 1e2, 1e5):
            m = GreedyOPM(
                r=4, s=12, p=3, mu=mu, grad_dilation=0.2,
                gamma_sp=0.0, ortho_method="arnoldi",
            ).fit(data)
            certs.append(m.lipschitz_certificate(dilation=0.2))
        assert certs[1] <= certs[0] * (1 + 1e-9)
        assert certs[2] <= certs[1] * (1 + 1e-9)
        # mu -> infinity drives the certificate to 1 (pure POD)
        assert certs[2] < certs[0] or np.isclose(certs[2], 1.0, atol=1e-6)

    def test_bounded_accuracy_loss(self, data):
        """Training error grows only mildly for moderate mu (Lemma 2.10).

        The natural scale of mu is n / max_k E_k: then mu * E_k <= n and the
        per-feature shrinkage n / (n + gamma + mu E_k) stays >= 1/2.
        """
        m0 = GreedyOPM(r=4, s=12, p=2, mu=0.0).fit(data)
        n = data.shape[1]
        m_probe = GreedyOPM(r=4, s=12, p=2, mu=1e-12).fit(data)
        mu = 0.5 * n / float(jnp.max(m_probe.dirichlet_nl_))
        m1 = GreedyOPM(r=4, s=12, p=2, mu=mu, grad_dilation=0.0).fit(data)
        e0 = m0.reconstruction_error(data)
        e1 = m1.reconstruction_error(data)
        assert e1 < 3.0 * max(e0, 1e-12) + 1e-6

    def test_certificate_matches_pointwise_norms(self, data):
        """Certificate upper-bounds the pointwise Jacobian norms."""
        m = GreedyOPM(r=3, s=10, p=2, mu=1e1, grad_dilation=0.1).fit(data)
        pts = (1.0 + 0.1) * m.S_hat_
        pointwise = np.asarray(m.decoder_jacobian_norms(pts))
        cert = m.lipschitz_certificate(dilation=0.1)
        assert float(pointwise.max()) <= cert * (1 + 1e-9)

    def test_ood_smoke(self, data):
        """Stabilized decode stays finite and controlled out of distribution."""
        S_train = data[:, :100]                     # t in [0, ~0.56]
        S_test = traveling_wave(n_t=160, t_max=0.9)[:, 100:]

        m_plain = GreedyOPM(r=4, s=12, p=3, mu=0.0).fit(S_train)
        m_stab = GreedyOPM(
            r=4, s=12, p=3, mu=1e3, grad_dilation=0.25, n_grad_samples=40,
        ).fit(S_train)

        err_plain = m_plain.reconstruction_error(S_test)
        err_stab = m_stab.reconstruction_error(S_test)
        assert np.isfinite(err_stab)
        # the stabilized manifold must not be (much) worse OOD
        assert err_stab <= max(2.0 * err_plain, 1.5)

    def test_stability_aware_activation_subset(self, data):
        """Weighted rule activates a subset of the plain rule (same basis)."""
        m = GreedyOPM(
            r=4, s=12, p=3, mu=1e-2, gamma_sp=1e-3, grad_dilation=0.1,
        ).fit(data)
        act_plain = set(np.asarray(m.active_nl_global_).tolist())
        # toggle on the SAME fitted model: identical basis, re-threshold only
        m.stability_aware_activation = True
        m.set_sparsity(m.gamma_sp)
        act_weighted = set(np.asarray(m.active_nl_global_).tolist())
        assert act_weighted <= act_plain

    def test_stability_aware_activation_warns_without_mu(self):
        with pytest.warns(UserWarning, match="stability_aware_activation"):
            GreedyOPM(r=2, s=6, p=2, mu=0.0, stability_aware_activation=True)

    def test_probe_subsampling(self, data):
        m = GreedyOPM(r=3, s=10, p=2, mu=1.0, n_grad_samples=25).fit(data)
        assert m.dirichlet_nl_.shape[0] == m.m_nl_
        assert np.all(np.asarray(m.dirichlet_nl_) >= 0.0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
