"""Tests for Legendre, Chebyshev, and Hermite polynomial implementations.

Verifies:
  1. Recurrence-relation correctness against known analytic values.
  2. Orthonormality under each family's weight function (numerical quadrature).
  3. Normalisation constants match the documented conventions.
  4. Generates diagnostic plots saved to  tests/plots/  for visual inspection.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

from greedyopm.polynomials.classical.legendre import _legendre_1d
from greedyopm.polynomials.classical.chebyshev import _chebyshev_1d
from greedyopm.polynomials.classical.hermite import _hermite_1d

# ---------------------------------------------------------------------------
# Output directory for plots
# ---------------------------------------------------------------------------
PLOT_DIR = Path(__file__).parent / "plots"


@pytest.fixture(autouse=True, scope="session")
def _create_plot_dir():
    """Ensure the plots/ directory exists before any test runs."""
    PLOT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

MAX_DEG = 5          # highest degree to test
N_QUAD = 20_000      # quadrature points for orthonormality checks


def gauss_legendre_quad(n):
    """Return (nodes, weights) for n-point Gauss-Legendre on [-1, 1]."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return jnp.array(nodes), jnp.array(weights)


# ===================================================================
#  LEGENDRE
# ===================================================================

class TestLegendre:
    """Normalised Legendre polynomials  P̃_n = sqrt(2n+1) P_n.

    Orthonormality:  ∫_{-1}^{1} P̃_i(x) P̃_j(x) dx/2 = δ_{ij}
    """

    # --- analytic spot-checks (unnormalised P_n) ---

    def test_P0_is_one(self):
        x = jnp.linspace(-1, 1, 50)
        vals = _legendre_1d(x, max_deg=0)  # (1, 50)
        # P̃_0 = sqrt(1) * 1 = 1
        np.testing.assert_allclose(vals[0], 1.0, atol=1e-7)

    def test_P1_is_sqrt3_x(self):
        x = jnp.linspace(-1, 1, 50)
        vals = _legendre_1d(x, max_deg=1)
        # P̃_1 = sqrt(3) x
        np.testing.assert_allclose(vals[1], jnp.sqrt(3.0) * x, atol=1e-6)

    def test_P2_at_zero(self):
        """P_2(0) = -1/2, so P̃_2(0) = sqrt(5) * (-1/2)."""
        vals = _legendre_1d(jnp.array([0.0]), max_deg=2)
        expected = jnp.sqrt(5.0) * (-0.5)
        np.testing.assert_allclose(vals[2, 0], expected, atol=1e-6)

    def test_P3_at_one(self):
        """P_n(1) = 1 for all n, so P̃_n(1) = sqrt(2n+1)."""
        vals = _legendre_1d(jnp.array([1.0]), max_deg=MAX_DEG)
        for n in range(MAX_DEG + 1):
            expected = jnp.sqrt(2.0 * n + 1.0)
            np.testing.assert_allclose(vals[n, 0], expected, atol=1e-6,
                                       err_msg=f"P̃_{n}(1) failed")

    # --- orthonormality ---

    def test_orthonormality(self):
        """∫_{-1}^{1} P̃_i P̃_j dx/2 = δ_{ij}  via Gauss-Legendre quadrature.

        Tolerance is set to 1e-5 to accommodate JAX's default float32.
        The quadrature itself is exact for polynomials of degree ≤ 2N-1;
        the residual comes from single-precision accumulation over 20k terms.
        """
        nodes, weights = gauss_legendre_quad(N_QUAD)
        vals = _legendre_1d(nodes, max_deg=MAX_DEG)  # (D+1, N)
        # Gram matrix  G_{ij} = sum_k w_k P̃_i(x_k) P̃_j(x_k) / 2
        G = (vals @ jnp.diag(weights) @ vals.T) / 2.0
        np.testing.assert_allclose(G, jnp.eye(MAX_DEG + 1), atol=1e-5)

    # --- plot ---

    def test_plot(self):
        x = jnp.linspace(-1, 1, 500)
        vals = _legendre_1d(x, max_deg=MAX_DEG)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: polynomial curves
        ax = axes[0]
        for n in range(MAX_DEG + 1):
            ax.plot(x, vals[n], label=rf"$\tilde{{P}}_{n}$")
        ax.set_title("Normalised Legendre polynomials")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\tilde{P}_n(x)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)

        # Right: Gram matrix heatmap (orthonormality check)
        nodes, weights = gauss_legendre_quad(N_QUAD)
        v = _legendre_1d(nodes, max_deg=MAX_DEG)
        G = (v @ jnp.diag(weights) @ v.T) / 2.0

        ax = axes[1]
        im = ax.imshow(jnp.abs(G), vmin=0, vmax=1.1, cmap="viridis")
        ax.set_title(r"$|G_{ij}|$  (should be identity)")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        for i in range(MAX_DEG + 1):
            for j in range(MAX_DEG + 1):
                ax.text(j, i, f"{float(G[i, j]):.2f}", ha="center", va="center",
                        fontsize=7, color="w" if float(jnp.abs(G[i, j])) > 0.5 else "k")
        fig.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle("Legendre polynomials — verification", fontsize=13)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "legendre.png", dpi=150)
        plt.close(fig)


# ===================================================================
#  CHEBYSHEV
# ===================================================================

class TestChebyshev:
    """Normalised Chebyshev (1st kind)  T̃_n.

    Weight  w(x) = 1 / (π sqrt(1 − x²)).
    Orthonormality:  ∫_{-1}^{1} T̃_i T̃_j w(x) dx = δ_{ij}.
    T̃_0 = 1,  T̃_n = sqrt(2) T_n  for n ≥ 1.
    """

    def test_T0_is_one(self):
        x = jnp.linspace(-0.99, 0.99, 50)
        vals = _chebyshev_1d(x, max_deg=0)
        np.testing.assert_allclose(vals[0], 1.0, atol=1e-7)

    def test_T1_is_sqrt2_x(self):
        x = jnp.linspace(-0.99, 0.99, 50)
        vals = _chebyshev_1d(x, max_deg=1)
        # T̃_1 = sqrt(2) x
        np.testing.assert_allclose(vals[1], jnp.sqrt(2.0) * x, atol=1e-6)

    def test_T2_formula(self):
        """T_2(x) = 2x² − 1,  T̃_2 = sqrt(2)(2x² − 1)."""
        x = jnp.linspace(-0.99, 0.99, 50)
        vals = _chebyshev_1d(x, max_deg=2)
        expected = jnp.sqrt(2.0) * (2.0 * x**2 - 1.0)
        np.testing.assert_allclose(vals[2], expected, atol=1e-5)

    def test_T3_formula(self):
        """T_3(x) = 4x³ − 3x,  T̃_3 = sqrt(2)(4x³ − 3x)."""
        x = jnp.linspace(-0.99, 0.99, 50)
        vals = _chebyshev_1d(x, max_deg=3)
        expected = jnp.sqrt(2.0) * (4.0 * x**3 - 3.0 * x)
        np.testing.assert_allclose(vals[3], expected, atol=1e-5)

    def test_orthonormality(self):
        """Numerical quadrature with Chebyshev weight  w = 1/(π√(1−x²))."""
        # Use Gauss-Chebyshev quadrature: nodes = cos((2k-1)π / 2N), weights = π/N
        N = N_QUAD
        k = jnp.arange(1, N + 1)
        nodes = jnp.cos((2 * k - 1) * jnp.pi / (2 * N))
        weights = jnp.ones(N) * jnp.pi / N
        # Gauss-Chebyshev integrates ∫ f(x)/sqrt(1-x^2) dx = sum w_k f(x_k)
        # We want ∫ T̃_i T̃_j / (π sqrt(1-x^2)) dx = (1/π) sum w_k T̃_i T̃_j
        vals = _chebyshev_1d(nodes, max_deg=MAX_DEG)
        G = (1.0 / jnp.pi) * (vals @ jnp.diag(weights) @ vals.T)
        # Chebyshev has slightly larger float32 error due to the weight function
        np.testing.assert_allclose(G, jnp.eye(MAX_DEG + 1), atol=1e-4)

    def test_plot(self):
        x = jnp.linspace(-1, 1, 500)
        vals = _chebyshev_1d(x, max_deg=MAX_DEG)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for n in range(MAX_DEG + 1):
            ax.plot(x, vals[n], label=rf"$\tilde{{T}}_{n}$")
        ax.set_title("Normalised Chebyshev polynomials (1st kind)")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\tilde{T}_n(x)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)

        # Gram matrix with Chebyshev weight
        N = N_QUAD
        k = jnp.arange(1, N + 1)
        nodes = jnp.cos((2 * k - 1) * jnp.pi / (2 * N))
        weights = jnp.ones(N) * jnp.pi / N
        v = _chebyshev_1d(nodes, max_deg=MAX_DEG)
        G = (1.0 / jnp.pi) * (v @ jnp.diag(weights) @ v.T)

        ax = axes[1]
        im = ax.imshow(jnp.abs(G), vmin=0, vmax=1.1, cmap="viridis")
        ax.set_title(r"$|G_{ij}|$  (should be identity)")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        for i in range(MAX_DEG + 1):
            for j in range(MAX_DEG + 1):
                ax.text(j, i, f"{float(G[i, j]):.2f}", ha="center", va="center",
                        fontsize=7, color="w" if float(jnp.abs(G[i, j])) > 0.5 else "k")
        fig.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle("Chebyshev polynomials — verification", fontsize=13)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "chebyshev.png", dpi=150)
        plt.close(fig)


# ===================================================================
#  HERMITE
# ===================================================================

class TestHermite:
    """Normalised probabilist's Hermite  H̃e_n = He_n / sqrt(n!).

    Weight  w(x) = exp(−x²/2) / sqrt(2π).
    Orthonormality:  ∫ H̃e_i H̃e_j w(x) dx = δ_{ij}.
    """

    def test_He0_is_one(self):
        x = jnp.linspace(-4, 4, 50)
        vals = _hermite_1d(x, max_deg=0)
        np.testing.assert_allclose(vals[0], 1.0, atol=1e-7)

    def test_He1_is_x(self):
        """He_1(x) = x, normalised: H̃e_1 = x / sqrt(1!) = x."""
        x = jnp.linspace(-4, 4, 50)
        vals = _hermite_1d(x, max_deg=1)
        np.testing.assert_allclose(vals[1], x, atol=1e-6)

    def test_He2_formula(self):
        """He_2(x) = x² − 1,  H̃e_2 = (x² − 1) / sqrt(2)."""
        x = jnp.linspace(-4, 4, 50)
        vals = _hermite_1d(x, max_deg=2)
        expected = (x**2 - 1.0) / jnp.sqrt(2.0)
        np.testing.assert_allclose(vals[2], expected, atol=1e-5)

    def test_He3_formula(self):
        """He_3(x) = x³ − 3x,  H̃e_3 = (x³ − 3x) / sqrt(6)."""
        x = jnp.linspace(-4, 4, 50)
        vals = _hermite_1d(x, max_deg=3)
        expected = (x**3 - 3.0 * x) / jnp.sqrt(6.0)
        np.testing.assert_allclose(vals[3], expected, atol=1e-5)

    def test_He4_formula(self):
        """He_4(x) = x⁴ − 6x² + 3,  H̃e_4 = (x⁴ − 6x² + 3) / sqrt(24)."""
        x = jnp.linspace(-4, 4, 50)
        vals = _hermite_1d(x, max_deg=4)
        expected = (x**4 - 6.0 * x**2 + 3.0) / jnp.sqrt(24.0)
        np.testing.assert_allclose(vals[4], expected, atol=1e-4)

    def test_orthonormality(self):
        """∫ H̃e_i H̃e_j exp(−x²/2)/sqrt(2π) dx = δ_{ij}
        via Gauss-Hermite quadrature (probabilist's).
        """
        # numpy's hermgauss uses the physicist convention  w = exp(-x^2).
        # Transform:  ∫ f(x) exp(-x^2/2)/sqrt(2pi) dx
        #           = ∫ f(sqrt(2) t) exp(-t^2) dt / sqrt(pi)
        N = 200  # Gauss-Hermite nodes are sufficient at moderate count
        nodes_phys, weights_phys = np.polynomial.hermite_e.hermegauss(N)
        # hermegauss gives nodes/weights for ∫ f(x) exp(-x^2/2) dx
        # so we just need to divide by sqrt(2pi) for our normalisation.
        nodes = jnp.array(nodes_phys)
        weights = jnp.array(weights_phys) / jnp.sqrt(2.0 * jnp.pi)

        vals = _hermite_1d(nodes, max_deg=MAX_DEG)
        G = vals @ jnp.diag(weights) @ vals.T
        np.testing.assert_allclose(G, jnp.eye(MAX_DEG + 1), atol=1e-5)

    def test_plot(self):
        x = jnp.linspace(-4, 4, 500)
        vals = _hermite_1d(x, max_deg=MAX_DEG)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for n in range(MAX_DEG + 1):
            ax.plot(x, vals[n], label=rf"$\tilde{{He}}_{n}$")
        ax.set_title("Normalised probabilist's Hermite polynomials")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\tilde{He}_n(x)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)

        # Gram matrix with Gaussian weight
        N = 200
        nodes_phys, weights_phys = np.polynomial.hermite_e.hermegauss(N)
        nodes_q = jnp.array(nodes_phys)
        weights_q = jnp.array(weights_phys) / jnp.sqrt(2.0 * jnp.pi)
        v = _hermite_1d(nodes_q, max_deg=MAX_DEG)
        G = v @ jnp.diag(weights_q) @ v.T

        ax = axes[1]
        im = ax.imshow(jnp.abs(G), vmin=0, vmax=1.1, cmap="viridis")
        ax.set_title(r"$|G_{ij}|$  (should be identity)")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        for i in range(MAX_DEG + 1):
            for j in range(MAX_DEG + 1):
                ax.text(j, i, f"{float(G[i, j]):.2f}", ha="center", va="center",
                        fontsize=7, color="w" if float(jnp.abs(G[i, j])) > 0.5 else "k")
        fig.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle("Hermite polynomials — verification", fontsize=13)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "hermite.png", dpi=150)
        plt.close(fig)


# ===================================================================
#  COMBINED COMPARISON PLOT
# ===================================================================

class TestCombinedPlot:
    """Generates a single figure comparing all three families side by side."""

    def test_combined_plot(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        families = [
            ("Legendre", _legendre_1d, jnp.linspace(-1, 1, 500), r"$\tilde{P}_n$"),
            ("Chebyshev", _chebyshev_1d, jnp.linspace(-1, 1, 500), r"$\tilde{T}_n$"),
            ("Hermite", _hermite_1d, jnp.linspace(-4, 4, 500), r"$\tilde{He}_n$"),
        ]

        colors = plt.cm.tab10.colors

        for col, (name, fn, x, ylabel) in enumerate(families):
            vals = fn(x, max_deg=MAX_DEG)

            # Top row: polynomial curves
            ax = axes[0, col]
            for n in range(MAX_DEG + 1):
                ax.plot(x, vals[n], label=f"n={n}", color=colors[n], linewidth=1.5)
            ax.set_title(f"{name}", fontsize=13)
            ax.set_xlabel("x")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="k", linewidth=0.5)

            # Bottom row: Gram matrix
            if name == "Legendre":
                nodes, weights = gauss_legendre_quad(N_QUAD)
                v = fn(nodes, max_deg=MAX_DEG)
                G = (v @ jnp.diag(weights) @ v.T) / 2.0
            elif name == "Chebyshev":
                N = N_QUAD
                k = jnp.arange(1, N + 1)
                nodes = jnp.cos((2 * k - 1) * jnp.pi / (2 * N))
                weights = jnp.ones(N) * jnp.pi / N
                v = fn(nodes, max_deg=MAX_DEG)
                G = (1.0 / jnp.pi) * (v @ jnp.diag(weights) @ v.T)
            else:  # Hermite
                nodes_p, weights_p = np.polynomial.hermite_e.hermegauss(200)
                nodes = jnp.array(nodes_p)
                weights = jnp.array(weights_p) / jnp.sqrt(2.0 * jnp.pi)
                v = fn(nodes, max_deg=MAX_DEG)
                G = v @ jnp.diag(weights) @ v.T

            ax = axes[1, col]
            im = ax.imshow(jnp.abs(G), vmin=0, vmax=1.1, cmap="viridis")
            ax.set_title(f"{name} Gram matrix |G_{{ij}}|")
            ax.set_xlabel("j")
            ax.set_ylabel("i")
            for i in range(MAX_DEG + 1):
                for j in range(MAX_DEG + 1):
                    ax.text(j, i, f"{float(G[i, j]):.2f}", ha="center",
                            va="center", fontsize=7,
                            color="w" if float(jnp.abs(G[i, j])) > 0.5 else "k")
            fig.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(
            "Orthogonal polynomial families — curves and orthonormality verification",
            fontsize=15,
        )
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "all_families.png", dpi=150)
        plt.close(fig)


# ===================================================================
#  EDGE CASES
# ===================================================================

class TestEdgeCases:
    """Boundary and degenerate inputs."""

    @pytest.mark.parametrize("fn", [_legendre_1d, _chebyshev_1d, _hermite_1d])
    def test_max_deg_zero(self, fn):
        x = jnp.linspace(-1, 1, 10)
        vals = fn(x, max_deg=0)
        assert vals.shape == (1, 10)
        np.testing.assert_allclose(vals[0], 1.0, atol=1e-7)

    @pytest.mark.parametrize("fn", [_legendre_1d, _chebyshev_1d, _hermite_1d])
    def test_scalar_input(self, fn):
        val = fn(jnp.array(0.5), max_deg=3)
        assert val.shape == (4, 1)

    @pytest.mark.parametrize("fn", [_legendre_1d, _chebyshev_1d, _hermite_1d])
    def test_high_degree(self, fn):
        """Smoke test: degree 20 should not NaN."""
        x = jnp.linspace(-0.9, 0.9, 100)
        vals = fn(x, max_deg=20)
        assert vals.shape == (21, 100)
        assert jnp.all(jnp.isfinite(vals))
