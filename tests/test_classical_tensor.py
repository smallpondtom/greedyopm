"""Tests for evaluate_classical_tensor.

Verifies:
  1. Output shape is (m, n) for inputs S_hat (r, n) and multi_indices (m, r).
  2. In the 1D case (r=1) the result matches direct univariate evaluation.
  3. In the 2D case individual tensor-product values match manual products of
     univariate evaluations.
  4. All three families produce finite, correctly-shaped output.
  5. Generates diagnostic plots in tests/plots/ for visual inspection.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from greedyopm.polynomials.classical_tensor import evaluate_classical_tensor
from greedyopm.polynomials.legendre import _legendre_1d
from greedyopm.polynomials.chebyshev import _chebyshev_1d
from greedyopm.polynomials.hermite import _hermite_1d

# ---------------------------------------------------------------------------
PLOT_DIR = Path(__file__).parent / "plots"
MAX_DEG = 4
N_PTS = 80   # number of sample points along each axis


@pytest.fixture(autouse=True, scope="session")
def _create_plot_dir():
    PLOT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_multi_indices_1d(max_deg: int) -> jnp.ndarray:
    """Return shape-(max_deg+1, 1) array of 1-D multi-indices [[0],[1],...,[D]]."""
    return jnp.arange(max_deg + 1, dtype=jnp.int32)[:, None]


def _canonical_2d_indices() -> jnp.ndarray:
    """A small set of 2-D multi-indices for manual verification."""
    return jnp.array(
        [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [2, 1], [1, 2]],
        dtype=jnp.int32,
    )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestShape:
    @pytest.mark.parametrize("family", ["legendre", "chebyshev", "hermite"])
    def test_output_shape_1d(self, family):
        r, n, m = 1, 50, MAX_DEG + 1
        S_hat = jnp.linspace(-0.9, 0.9, n)[None, :]   # (1, n)
        mi = _all_multi_indices_1d(MAX_DEG)             # (m, 1)
        out = evaluate_classical_tensor(S_hat, mi, family, MAX_DEG)
        assert out.shape == (m, n), f"expected ({m},{n}), got {out.shape}"

    @pytest.mark.parametrize("family", ["legendre", "chebyshev", "hermite"])
    def test_output_shape_2d(self, family):
        r, n = 2, 60
        S_hat = jax.random.normal(jax.random.PRNGKey(0), (r, n))
        mi = _canonical_2d_indices()   # (8, 2)
        out = evaluate_classical_tensor(S_hat, mi, family, MAX_DEG)
        assert out.shape == (8, n)

    @pytest.mark.parametrize("family", ["legendre", "chebyshev", "hermite"])
    def test_all_finite(self, family):
        r, n = 3, 40
        key = jax.random.PRNGKey(42)
        S_hat = jax.random.uniform(key, (r, n), minval=-1.0, maxval=1.0)
        mi = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                         [2, 1, 0], [1, 1, 1]], dtype=jnp.int32)
        out = evaluate_classical_tensor(S_hat, mi, family, MAX_DEG)
        assert jnp.all(jnp.isfinite(out)), "output contains NaN or Inf"


# ---------------------------------------------------------------------------
# 1-D consistency: tensor result should equal the univariate function
# ---------------------------------------------------------------------------

class TestUnivariate:
    """For r=1, evaluate_classical_tensor must match direct 1-D evaluation."""

    def _run(self, family: str, uni_fn, x_range):
        x = jnp.linspace(*x_range, N_PTS)
        S_hat = x[None, :]                              # (1, N_PTS)
        mi = _all_multi_indices_1d(MAX_DEG)             # (D+1, 1)

        out = evaluate_classical_tensor(S_hat, mi, family, MAX_DEG)  # (D+1, N_PTS)

        # Direct univariate evaluation (no rescaling needed — same domain)
        # The function rescales internally, so we replicate that rescaling here.
        if family == "hermite":
            mu = jnp.mean(x)
            sigma = jnp.maximum(jnp.std(x), 1e-15)
            xi = (x - mu) / sigma
        else:
            a, b = jnp.min(x), jnp.max(x)
            rng = jnp.maximum(b - a, 1e-15)
            xi = 2.0 * (x - a) / rng - 1.0
            if family == "chebyshev":
                xi = xi * 0.999

        expected = uni_fn(xi, max_deg=MAX_DEG)   # (D+1, N_PTS)
        np.testing.assert_allclose(out, expected, atol=1e-5,
                                   err_msg=f"1-D mismatch for {family}")

    def test_legendre_1d(self):
        self._run("legendre", _legendre_1d, (-1.0, 1.0))

    def test_chebyshev_1d(self):
        self._run("chebyshev", _chebyshev_1d, (-0.9, 0.9))

    def test_hermite_1d(self):
        self._run("hermite", _hermite_1d, (-3.0, 3.0))


# ---------------------------------------------------------------------------
# 2-D tensor product: output[i, j] == prod_d uni_vals[d, alpha_d, j]
# ---------------------------------------------------------------------------

class TestTensorProduct2D:
    """Manually compute tensor products and compare."""

    def _rescale(self, S_hat: jnp.ndarray, family: str):
        """Replicate the coordinate rescaling done inside evaluate_classical_tensor."""
        if family == "hermite":
            mu = jnp.mean(S_hat, axis=1, keepdims=True)
            sigma = jnp.maximum(jnp.std(S_hat, axis=1, keepdims=True), 1e-15)
            return (S_hat - mu) / sigma
        else:
            a = jnp.min(S_hat, axis=1, keepdims=True)
            b = jnp.max(S_hat, axis=1, keepdims=True)
            rng = jnp.maximum(b - a, 1e-15)
            xi = 2.0 * (S_hat - a) / rng - 1.0
            if family == "chebyshev":
                xi = xi * 0.999
            return xi

    def _manual_tensor(self, S_hat, mi, family, uni_fn):
        xi = self._rescale(S_hat, family)   # (r, n)
        r, n = xi.shape
        # uni_vals[d, deg, j]
        uni_vals = jnp.stack([uni_fn(xi[d], max_deg=MAX_DEG) for d in range(r)], axis=0)
        # (r, D+1, n)
        results = []
        for alpha in mi:
            prod = jnp.ones(n)
            for d, deg in enumerate(alpha):
                prod = prod * uni_vals[d, deg, :]
            results.append(prod)
        return jnp.stack(results, axis=0)   # (m, n)

    @pytest.mark.parametrize("family,uni_fn", [
        ("legendre", _legendre_1d),
        ("chebyshev", _chebyshev_1d),
        ("hermite", _hermite_1d),
    ])
    def test_2d_tensor(self, family, uni_fn):
        key = jax.random.PRNGKey(7)
        S_hat = jax.random.uniform(key, (2, N_PTS), minval=-0.8, maxval=0.8)
        mi = _canonical_2d_indices()

        out = evaluate_classical_tensor(S_hat, mi, family, MAX_DEG)
        expected = self._manual_tensor(S_hat, mi, family, uni_fn)
        np.testing.assert_allclose(out, expected, atol=1e-5,
                                   err_msg=f"2-D tensor mismatch for {family}")


# ---------------------------------------------------------------------------
# Plot: 1-D basis functions comparison
# ---------------------------------------------------------------------------

class TestPlot1D:
    """Plot all basis functions for r=1 and compare with univariate evaluation."""

    @pytest.mark.parametrize("family,uni_fn,x_range", [
        ("legendre", _legendre_1d, (-1.0, 1.0)),
        ("chebyshev", _chebyshev_1d, (-0.9, 0.9)),
        ("hermite", _hermite_1d, (-3.5, 3.5)),
    ])
    def test_plot_1d_basis(self, family, uni_fn, x_range):
        x = jnp.linspace(*x_range, 300)
        S_hat = x[None, :]                         # (1, 300)
        mi = _all_multi_indices_1d(MAX_DEG)        # (D+1, 1)

        out = evaluate_classical_tensor(S_hat, mi, family, MAX_DEG)  # (D+1, 300)

        # Rescale for reference
        if family == "hermite":
            mu = jnp.mean(x)
            sigma = jnp.maximum(jnp.std(x), 1e-15)
            xi = (x - mu) / sigma
        else:
            a, b = jnp.min(x), jnp.max(x)
            rng = jnp.maximum(b - a, 1e-15)
            xi = 2.0 * (x - a) / rng - 1.0
            if family == "chebyshev":
                xi = xi * 0.999
        ref = uni_fn(xi, max_deg=MAX_DEG)          # (D+1, 300)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = plt.cm.tab10.colors

        # Left: basis functions from evaluate_classical_tensor
        ax = axes[0]
        for n in range(MAX_DEG + 1):
            ax.plot(x, out[n], color=colors[n], linewidth=2, label=f"deg={n}")
        ax.set_title(f"{family.capitalize()} — evaluate_classical_tensor (r=1)")
        ax.set_xlabel("x (original domain)")
        ax.set_ylabel("basis value")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)

        # Right: residual |out - ref|
        ax = axes[1]
        for n in range(MAX_DEG + 1):
            ax.semilogy(x, jnp.abs(out[n] - ref[n]) + 1e-15,
                        color=colors[n], linewidth=1.5, label=f"deg={n}")
        ax.set_title(f"|evaluate_classical_tensor − _1d reference|")
        ax.set_xlabel("x")
        ax.set_ylabel("absolute error")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1e-9, 1e-2)

        fig.suptitle(
            f"1-D consistency check — {family.capitalize()} (r=1, max_deg={MAX_DEG})",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"classical_tensor_1d_{family}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: 2-D contour plots showing tensor-product structure
# ---------------------------------------------------------------------------

class TestPlot2D:
    """2-D basis function contour plots — one panel per multi-index."""

    @pytest.mark.parametrize("family", ["legendre", "chebyshev", "hermite"])
    def test_plot_2d_contours(self, family):
        # Build a 2-D grid
        if family == "hermite":
            grid_1d = np.linspace(-3.0, 3.0, N_PTS)
        else:
            grid_1d = np.linspace(-0.9, 0.9, N_PTS)

        X1, X2 = np.meshgrid(grid_1d, grid_1d)   # each (N, N)
        n_grid = N_PTS * N_PTS
        # S_hat: (2, n_grid)  — row 0 is x1, row 1 is x2
        S_hat = jnp.array(
            np.stack([X1.ravel(), X2.ravel()], axis=0), dtype=jnp.float32
        )

        mi = _canonical_2d_indices()   # (8, 2)
        out = evaluate_classical_tensor(S_hat, mi, family, MAX_DEG)  # (8, n_grid)

        mi_np = np.array(mi)
        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        axes_flat = axes.ravel()

        for idx in range(len(mi_np)):
            Z = np.array(out[idx]).reshape(N_PTS, N_PTS)
            alpha = tuple(int(a) for a in mi_np[idx])
            ax = axes_flat[idx]
            vmax = max(abs(Z.max()), abs(Z.min()), 0.01)
            cf = ax.contourf(X1, X2, Z, levels=20, cmap="RdBu_r",
                             vmin=-vmax, vmax=vmax)
            ax.contour(X1, X2, Z, levels=[0.0], colors="k", linewidths=0.8,
                       linestyles="--")
            fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
            ax.set_title(rf"$\alpha={alpha}$", fontsize=11)
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")

        fig.suptitle(
            f"Tensor-product {family.capitalize()} basis — 2-D contour plots\n"
            f"(r=2, max_deg={MAX_DEG})",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"classical_tensor_2d_{family}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: basis matrix Phi_tilde heatmap (snapshot × basis)
# ---------------------------------------------------------------------------

class TestPlotBasisMatrix:
    """Show Phi_tilde as a heatmap: rows = multi-indices, cols = snapshots."""

    @pytest.mark.parametrize("family", ["legendre", "chebyshev", "hermite"])
    def test_plot_basis_matrix(self, family):
        r, n = 2, 40
        key = jax.random.PRNGKey(99)
        S_hat = jax.random.uniform(key, (r, n), minval=-0.8, maxval=0.8)

        # Use all multi-indices up to total degree 2 (for a readable heatmap)
        total_deg = 2
        mi_list = [
            [i, j]
            for i in range(total_deg + 1)
            for j in range(total_deg + 1 - i)
        ]
        mi = jnp.array(mi_list, dtype=jnp.int32)   # (m, 2)

        out = evaluate_classical_tensor(S_hat, mi, family, MAX_DEG)  # (m, n)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Left: heatmap of Phi_tilde
        ax = axes[0]
        im = ax.imshow(np.array(out), aspect="auto", cmap="RdBu_r",
                       vmin=-float(jnp.max(jnp.abs(out))),
                       vmax=float(jnp.max(jnp.abs(out))))
        ax.set_xlabel("Snapshot index j")
        ax.set_ylabel("Multi-index i")
        ax.set_yticks(range(len(mi_list)))
        ax.set_yticklabels([str(tuple(a)) for a in mi_list], fontsize=8)
        ax.set_title(r"$\tilde{\Phi}$ (basis matrix)")
        fig.colorbar(im, ax=ax, shrink=0.8)

        # Right: Gram matrix G = Phi_tilde @ Phi_tilde^T / n
        G = (out @ out.T) / n
        ax = axes[1]
        im2 = ax.imshow(np.array(jnp.abs(G)), cmap="viridis", aspect="auto")
        ax.set_xlabel("i")
        ax.set_ylabel("i'")
        ax.set_title(r"$|G|= |\tilde{\Phi}\tilde{\Phi}^T / n|$")
        ax.set_xticks(range(len(mi_list)))
        ax.set_xticklabels([str(tuple(a)) for a in mi_list], fontsize=7,
                           rotation=60, ha="right")
        ax.set_yticks(range(len(mi_list)))
        ax.set_yticklabels([str(tuple(a)) for a in mi_list], fontsize=7)
        fig.colorbar(im2, ax=ax, shrink=0.8)

        fig.suptitle(
            f"Basis matrix & empirical Gram — {family.capitalize()} "
            f"(r=2, total_deg≤{total_deg}, n={n} random snapshots)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"classical_tensor_matrix_{family}.png", dpi=150)
        plt.close(fig)
