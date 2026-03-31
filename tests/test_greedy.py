"""Tests for the GreedyOPM class."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from greedyopm import GreedyOPM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_quadratic_data():
    """Snapshot matrix living on a quadratic manifold with small noise.

    The data has the form:
        S = V_lin @ s_hat + V_quad @ [s1^2, s2^2, s1*s2] + noise
    so a degree-2 polynomial manifold should capture it well.
    """
    np.random.seed(42)
    n, k, r = 100, 200, 3

    s_hat = np.random.randn(r, k)
    V_lin = np.random.randn(n, r)
    V_quad = np.random.randn(n, 3) * 0.3
    quad_features = np.array([
        s_hat[0] ** 2,
        s_hat[1] ** 2,
        s_hat[0] * s_hat[1],
    ])
    S = V_lin @ s_hat + V_quad @ quad_features + 0.01 * np.random.randn(n, k)
    return jnp.array(S)


@pytest.fixture
def linear_data():
    """Purely linear data (rank-3) — nonlinear features should not help."""
    np.random.seed(123)
    n, k, r = 50, 100, 3
    U = np.linalg.qr(np.random.randn(n, r))[0]
    s_hat = np.random.randn(r, k)
    S = U @ s_hat + 0.001 * np.random.randn(n, k)
    return jnp.array(S)


# ---------------------------------------------------------------------------
# Basic fit / reconstruct
# ---------------------------------------------------------------------------

class TestFitAndReconstruct:

    def test_fit_returns_self(self, synthetic_quadratic_data):
        model = GreedyOPM(r=3, p=2)
        result = model.fit(synthetic_quadratic_data)
        assert result is model

    def test_fitted_flag(self, synthetic_quadratic_data):
        model = GreedyOPM(r=3, p=2)
        assert not model._fitted
        model.fit(synthetic_quadratic_data)
        assert model._fitted

    def test_shapes_after_fit(self, synthetic_quadratic_data):
        n, k = synthetic_quadratic_data.shape
        r, p = 3, 3
        model = GreedyOPM(r=r, p=p).fit(synthetic_quadratic_data)

        assert model.s_bar_.shape == (n, 1)
        assert model.U_r_.shape == (n, r)
        assert model.S_hat_.shape == (r, k)
        assert model.Psi_.shape[1] == k
        assert model.C_.shape == (n, model.Psi_.shape[0])

    def test_reconstruct_shape(self, synthetic_quadratic_data):
        model = GreedyOPM(r=3, p=2).fit(synthetic_quadratic_data)
        S_approx = model.reconstruct(synthetic_quadratic_data)
        assert S_approx.shape == synthetic_quadratic_data.shape

    def test_opm_improves_over_pod(self, synthetic_quadratic_data):
        """Polynomial manifold should beat linear POD on quadratic data."""
        model = GreedyOPM(r=3, p=2, family="legendre").fit(synthetic_quadratic_data)
        pod_err = model.linear_error(synthetic_quadratic_data)
        opm_err = model.reconstruction_error(synthetic_quadratic_data)
        assert opm_err < pod_err


# ---------------------------------------------------------------------------
# Greedy feature selection
# ---------------------------------------------------------------------------

class TestGreedySelection:

    def test_n_features_limits_selection(self, synthetic_quadratic_data):
        model = GreedyOPM(r=3, p=3).fit(synthetic_quadratic_data, n_features=5)
        assert model.n_selected_nonlinear == 5

    def test_n_features_zero(self, synthetic_quadratic_data):
        """Zero nonlinear features should give POD-level error."""
        model = GreedyOPM(r=3, p=2).fit(synthetic_quadratic_data, n_features=0)
        assert model.n_selected_nonlinear == 0
        opm_err = model.reconstruction_error(synthetic_quadratic_data)
        pod_err = model.linear_error(synthetic_quadratic_data)
        assert abs(opm_err - pod_err) < 1e-4

    def test_tolerance_selects_fewer_than_all(self, synthetic_quadratic_data):
        model_all = GreedyOPM(r=3, p=3).fit(synthetic_quadratic_data)
        model_tol = GreedyOPM(r=3, p=3).fit(synthetic_quadratic_data, tol=0.05)
        assert model_tol.n_selected_nonlinear <= model_all.n_selected_nonlinear

    def test_more_features_lower_error(self, synthetic_quadratic_data):
        model_few = GreedyOPM(r=3, p=3).fit(synthetic_quadratic_data, n_features=3)
        model_many = GreedyOPM(r=3, p=3).fit(synthetic_quadratic_data, n_features=10)
        err_few = model_few.reconstruction_error(synthetic_quadratic_data)
        err_many = model_many.reconstruction_error(synthetic_quadratic_data)
        assert err_many <= err_few + 1e-10

    def test_linear_data_no_nonlinear_benefit(self, linear_data):
        """On purely linear data, nonlinear features add negligible value."""
        model = GreedyOPM(r=3, p=2).fit(linear_data)
        pod_err = model.linear_error(linear_data)
        opm_err = model.reconstruction_error(linear_data)
        # OPM should not be dramatically better than POD
        assert opm_err < pod_err + 1e-3


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

class TestEncodeDecode:

    def test_single_snapshot(self, synthetic_quadratic_data):
        model = GreedyOPM(r=3, p=2).fit(synthetic_quadratic_data)
        s = synthetic_quadratic_data[:, 0]

        s_hat = model.encode(s)
        assert s_hat.shape == (3,)

        s_recon = model.decode(s_hat)
        assert s_recon.shape == s.shape

    def test_batch_snapshots(self, synthetic_quadratic_data):
        model = GreedyOPM(r=3, p=2).fit(synthetic_quadratic_data)
        S_hat = model.encode(synthetic_quadratic_data)
        assert S_hat.shape == (3, synthetic_quadratic_data.shape[1])

        S_recon = model.decode(S_hat)
        assert S_recon.shape == synthetic_quadratic_data.shape

    def test_encode_decode_matches_reconstruct(self, synthetic_quadratic_data):
        model = GreedyOPM(r=3, p=2).fit(synthetic_quadratic_data)
        via_reconstruct = model.reconstruct(synthetic_quadratic_data)
        via_encode_decode = model.decode(model.encode(synthetic_quadratic_data))
        np.testing.assert_allclose(via_reconstruct, via_encode_decode, atol=1e-6)


# ---------------------------------------------------------------------------
# Basis and method variants
# ---------------------------------------------------------------------------

class TestVariants:

    @pytest.mark.parametrize("family", ["legendre", "chebyshev", "hermite"])
    def test_all_families(self, synthetic_quadratic_data, family):
        model = GreedyOPM(r=3, p=2, family=family, basis="hybrid")
        model.fit(synthetic_quadratic_data, n_features=5)
        err = model.reconstruction_error(synthetic_quadratic_data)
        assert err < 0.5  # reasonable reconstruction

    def test_hierarchical_basis(self, synthetic_quadratic_data):
        model = GreedyOPM(r=3, p=2, basis="hierarchical")
        model.fit(synthetic_quadratic_data, n_features=5)
        err = model.reconstruction_error(synthetic_quadratic_data)
        assert err < 0.5

    @pytest.mark.parametrize("method", ["qr", "cholesky"])
    def test_ortho_methods(self, synthetic_quadratic_data, method):
        model = GreedyOPM(r=3, p=2, ortho_method=method)
        model.fit(synthetic_quadratic_data, n_features=5)
        err = model.reconstruction_error(synthetic_quadratic_data)
        assert err < 0.5

    def test_no_centering(self, synthetic_quadratic_data):
        model = GreedyOPM(r=3, p=2, center=False)
        model.fit(synthetic_quadratic_data)
        err = model.reconstruction_error(synthetic_quadratic_data)
        assert err < 1.0  # may be worse without centering but should not crash


# ---------------------------------------------------------------------------
# Orthonormality verification
# ---------------------------------------------------------------------------

class TestOrthonormality:

    def test_psi_orthonormal(self, synthetic_quadratic_data):
        """Verify (1/k) Ψ Ψᵀ ≈ I."""
        model = GreedyOPM(r=3, p=2).fit(synthetic_quadratic_data)
        k = synthetic_quadratic_data.shape[1]
        gram = (1.0 / k) * model.Psi_ @ model.Psi_.T
        np.testing.assert_allclose(gram, jnp.eye(gram.shape[0]), atol=1e-6)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:

    def test_energy_captured_sums(self, synthetic_quadratic_data):
        model = GreedyOPM(r=3, p=3).fit(synthetic_quadratic_data)
        energy = model.energy_captured()
        total = energy["pod"] + energy["nonlinear"] + energy["residual"]
        np.testing.assert_allclose(total, energy["total"], rtol=1e-3)

    def test_summary_keys(self, synthetic_quadratic_data):
        model = GreedyOPM(r=3, p=2).fit(synthetic_quadratic_data)
        s = model.summary()
        expected_keys = {
            "r", "p", "n_snapshots", "state_dim",
            "total_poly_features", "linear_features",
            "selected_nonlinear", "total_selected",
            "compression_ratio", "family", "basis", "ortho_method",
            "top_singular_values",
        }
        assert set(s.keys()) == expected_keys

    def test_compression_ratio(self, synthetic_quadratic_data):
        model_all = GreedyOPM(r=3, p=3).fit(synthetic_quadratic_data)
        assert model_all.compression_ratio == 1.0

        model_few = GreedyOPM(r=3, p=3).fit(synthetic_quadratic_data, n_features=2)
        assert model_few.compression_ratio < 1.0

    def test_repr(self, synthetic_quadratic_data):
        model = GreedyOPM(r=3, p=2)
        assert "not fitted" in repr(model)
        model.fit(synthetic_quadratic_data)
        assert "fitted" in repr(model)
