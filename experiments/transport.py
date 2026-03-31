"""
Compare Greedy-OPM vs Greedy-QM on the linear transport (advecting pulse) problem.

Generates an advecting Gaussian pulse, splits into train/test sets, and
compares reconstruction accuracy and timing of:
  - Linear POD (baseline)
  - Greedy Quadratic Manifold (QM)  [from experiments/qm/quadmani.py]
  - Greedy OPM with various polynomial degrees and families
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Make experiment modules importable
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "experiments"))

from data.transport import generate_advecting_pulse
from qm.quadmani import (
    quadmani_greedy,
    linear_reduce,
    lift_quadratic,
    default_feature_map,
)
from greedyopm import GreedyOPM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def relative_error(S_true, S_approx):
    """Relative Frobenius error  ||S - S_approx||_F / ||S||_F."""
    return float(jnp.linalg.norm(S_true - S_approx, "fro")
                 / jnp.linalg.norm(S_true, "fro"))


def pod_reconstruct(S_train, S_test, r, center=True):
    """Baseline: rank-r linear POD reconstruction of test data."""
    if center:
        s_bar = jnp.mean(S_train, axis=1, keepdims=True)
    else:
        s_bar = jnp.zeros((S_train.shape[0], 1))
    U, _, _ = jnp.linalg.svd(S_train - s_bar, full_matrices=False)
    U_r = U[:, :r]
    S_test_c = S_test - s_bar
    return s_bar + U_r @ (U_r.T @ S_test_c)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 70)
    print("  Greedy-OPM  vs  Greedy-QM  on the Linear Transport Problem")
    print("=" * 70)

    # --- Generate data ---
    print("\n[1/4] Generating advecting pulse data ...")
    S = generate_advecting_pulse(
        pulse_width=2e-4,
        pulse_shift=0.1,
        speed=10.0,
        final_time=0.1,
        n_time_samples=2000,
        n_space_samples=4096,
    )
    n, k = S.shape
    print("       Snapshot matrix: %d x %d  (n_space x n_time)" % (n, k))

    # Train/test split: even columns for training, odd for testing
    S_train = S[:, ::2]
    S_test = S[:, 1::2]
    k_train = S_train.shape[1]
    k_test = S_test.shape[1]
    print("       Train: %d snapshots,  Test: %d snapshots" % (k_train, k_test))

    # --- Sweep over reduced dimensions ---
    r_values = [3, 5, 7, 10]
    p_opm = 3  # max polynomial degree for OPM

    print("\n[2/4] Running POD baseline ...")
    pod_results = {}
    for r in r_values:
        S_recon = pod_reconstruct(S_train, S_test, r)
        pod_results[r] = relative_error(S_test, S_recon)
        print("       r=%2d  |  POD error = %.6e" % (r, pod_results[r]))

    # --- Greedy-QM ---
    print("\n[3/4] Running Greedy-QM (quadratic feature map) ...")
    qm_results = {}
    qm_times = {}
    for r in r_values:
        t0 = time.perf_counter()
        V, W, shift_val, idx_in = quadmani_greedy(
            S_train,
            r=r,
            n_vectors_to_check=min(200, k_train),
            reg_magnitude=1e-6,
            idx_in=jnp.array([], dtype=jnp.int32),
            feature_map=default_feature_map,
            shifting=True,
        )
        dt = time.perf_counter() - t0
        qm_times[r] = dt

        # Reconstruct test data
        reduced_test = linear_reduce(V, S_test, shift_val)
        S_recon = lift_quadratic(V, W, shift_val, reduced_test)
        qm_results[r] = relative_error(S_test, S_recon)
        print("       r=%2d  |  QM error = %.6e  (%.2f s)" % (r, qm_results[r], dt))

    # --- Greedy-OPM (multiple configurations) ---
    print("\n[4/4] Running Greedy-OPM ...")

    opm_configs = [
        {"family": "legendre", "basis": "hybrid",       "label": "OPM-Leg-hybrid"},
        {"family": "chebyshev", "basis": "hybrid",      "label": "OPM-Cheb-hybrid"},
        {"family": "hermite",  "basis": "hybrid",       "label": "OPM-Herm-hybrid"},
        {"family": "legendre", "basis": "hierarchical", "label": "OPM-monomial"},
    ]

    opm_results = {cfg["label"]: {} for cfg in opm_configs}
    opm_times   = {cfg["label"]: {} for cfg in opm_configs}

    for cfg in opm_configs:
        label = cfg["label"]
        print("       --- %s (p=%d) ---" % (label, p_opm))
        for r in r_values:
            t0 = time.perf_counter()
            model = GreedyOPM(
                r=r,
                p=p_opm,
                family=cfg["family"],
                basis=cfg["basis"],
                ortho_method="qr",
                center=True,
            )
            model.fit(S_train)  # keep all features (no pruning)
            dt = time.perf_counter() - t0
            opm_times[label][r] = dt

            # Reconstruct test data
            S_recon = model.reconstruct(S_test)
            err = relative_error(S_test, S_recon)
            opm_results[label][r] = err
            n_nl = model.n_selected_nonlinear
            print("       r=%2d  |  error = %.6e  (%d NL features, %.2f s)"
                  % (r, err, n_nl, dt))

    # --- Also test greedy-pruned OPM ---
    print("\n       --- OPM-Leg-hybrid GREEDY (n_features=r*(r+1)/2) ---")
    opm_greedy_results = {}
    opm_greedy_times = {}
    for r in r_values:
        # Match QM's feature count: r*(r+1)/2 quadratic features
        n_quad = r * (r + 1) // 2
        t0 = time.perf_counter()
        model = GreedyOPM(r=r, p=p_opm, family="legendre", basis="hybrid")
        model.fit(S_train, n_features=n_quad)
        dt = time.perf_counter() - t0
        opm_greedy_times[r] = dt

        S_recon = model.reconstruct(S_test)
        err = relative_error(S_test, S_recon)
        opm_greedy_results[r] = err
        print("       r=%2d  |  error = %.6e  (%d NL features, %.2f s)"
              % (r, err, model.n_selected_nonlinear, dt))

    # --- Summary table ---
    print("\n")
    print("=" * 70)
    print("  SUMMARY TABLE  (relative Frobenius test error)")
    print("=" * 70)

    header = "  r  |     POD      |    QM        |  OPM-full    | OPM-greedy   "
    print(header)
    print("-" * len(header))
    for r in r_values:
        print(" %2d  |  %.4e  |  %.4e  |  %.4e  |  %.4e"
              % (r,
                 pod_results[r],
                 qm_results[r],
                 opm_results["OPM-Leg-hybrid"][r],
                 opm_greedy_results[r]))

    print()
    print("  Timing (seconds)")
    print("-" * 50)
    header_t = "  r  |    QM     |  OPM-full | OPM-greedy"
    print(header_t)
    print("-" * len(header_t))
    for r in r_values:
        print(" %2d  |  %7.2f  |  %7.2f  |  %7.2f"
              % (r,
                 qm_times[r],
                 opm_times["OPM-Leg-hybrid"][r],
                 opm_greedy_times[r]))

    print("\n" + "=" * 70)
    print("  OPM family comparison (full features, test error)")
    print("=" * 70)
    header_f = "  r  |  Legendre    | Chebyshev    |  Hermite     |  Monomial    "
    print(header_f)
    print("-" * len(header_f))
    for r in r_values:
        print(" %2d  |  %.4e  |  %.4e  |  %.4e  |  %.4e"
              % (r,
                 opm_results["OPM-Leg-hybrid"][r],
                 opm_results["OPM-Cheb-hybrid"][r],
                 opm_results["OPM-Herm-hybrid"][r],
                 opm_results["OPM-monomial"][r]))

    print()


if __name__ == "__main__":
    run_experiment()
