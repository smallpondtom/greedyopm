"""Transport benchmark for Greedy-OPM and Greedy-Fixed-OPM.

Mirrors the user's ``experiments/transport.py`` template (trimmed to remove
the Greedy-QM/CM comparisons, which depend on a separate module).  Tests:

* POD baseline.
* GreedyFixedOPM with monomial / legendre / chebyshev / hermite families
  and each of qr / cholesky / arnoldi.
* GreedyOPM (sparse) with the same options.

For each we report train and test relative Frobenius errors and runtimes.
"""

#%%
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from data.transport import generate_advecting_pulse
from greedyopm import GreedyFixedOPM, GreedyOPM


def relative_error(S_true: jnp.ndarray, S_approx: jnp.ndarray) -> float:
    return float(
        jnp.linalg.norm(S_true - S_approx, "fro")
        / jnp.linalg.norm(S_true, "fro")
    )


def main(r: int = 10, s: int = 100, p: int = 2) -> None:
    print("=" * 72)
    print(f"  GreedyOPM transport benchmark  (r={r}, s={s}, p={p})")
    print("=" * 72)

    # ---- 1. Data ----
    print("\n[1/4] Generating advecting-pulse data ...")
    S, xspan, tspan = generate_advecting_pulse(
        pulse_width=5.0e-4,
        pulse_shift=0.1,
        speed=5.0,
        final_time=0.20,
        n_time_samples=2000,
        n_space_samples=2 ** 10,
    )
    n_space, n_time = S.shape
    print(f"     Snapshot matrix: {n_space} x {n_time}")

    S_train = S[:, ::2]
    S_test = S[:, 1::2]
    print(f"     Train: {S_train.shape[1]}  |  Test: {S_test.shape[1]}")

    # JIT warm-up: we compile the batched Arnoldi at each trial shape
    # (1, 2, ..., r).  Do a single cold-start fit so subsequent timings
    # reflect steady-state cost.
    print("\n[warmup] Cold-starting JIT caches ...")
    _ = GreedyOPM(
        r=r, s=s, p=p,
        gamma_sp=1e-4, gamma=1e-15, n_reorth=1,
        family="monomial", basis="hierarchical",
        ortho_method="arnoldi", center=True,
    ).fit(S_train)

    # ---- 2. POD baseline ----
    print("\n[2/4] Running POD baseline ...")
    s_ref = jnp.mean(S_train, axis=1, keepdims=True)
    U_pod, _, _ = jnp.linalg.svd(S_train - s_ref, full_matrices=False)
    U_r = U_pod[:, :r]
    S_pod_train = s_ref + U_r @ (U_r.T @ (S_train - s_ref))
    S_pod_test = s_ref + U_r @ (U_r.T @ (S_test - s_ref))
    print(f"     POD train error = {relative_error(S_train, S_pod_train):.6e}")
    print(f"     POD  test error = {relative_error(S_test,  S_pod_test ):.6e}")

    # ---- 3. GreedyFixedOPM sweep ----
    print("\n[3/4] Running GreedyFixedOPM ...")
    fixed_configs = [
        ("monomial",  "qr"),
        ("monomial",  "arnoldi"),
        ("legendre",  "qr"),
        ("hermite",   "qr"),
    ]
    for family, ortho in fixed_configs:
        t0 = time.perf_counter()
        model = GreedyFixedOPM(
            r=r, s=s, p=p,
            gamma=1e-14,
            n_reorth=2,
            family=family,
            ortho_method=ortho,
            center=True,
        )
        model.fit(S_train)
        S_tr = model.reconstruct(S_train)
        S_te = model.reconstruct(S_test)
        dt = time.perf_counter() - t0
        print(
            f"     family={family!r:11s}  ortho={ortho!r:10s}  "
            f"train={relative_error(S_train, S_tr):.3e}  "
            f"test={relative_error(S_test, S_te):.3e}  "
            f"[{dt:.2f}s]  modes={list(model.selected_modes)[:5]}..."
        )

    # ---- 4. GreedyOPM (sparse) sweep ----
    print("\n[4/4] Running GreedyOPM (sparse) ...")
    sparse_configs = [
        ("monomial", "arnoldi",      "hierarchical", 1e-4),
        ("monomial", "qr",           "hierarchical", 1e-4),
        ("legendre", "qr",           "hybrid",       1e-4),
    ]
    for family, ortho, basis, gamma_sp in sparse_configs:
        t0 = time.perf_counter()
        model = GreedyOPM(
            r=r, s=s, p=p,
            gamma_sp=gamma_sp,
            gamma=1e-15,
            n_reorth=2,
            family=family,
            basis=basis,
            ortho_method=ortho,
            center=True,
        )
        model.fit(S_train)
        S_tr = model.reconstruct(S_train)
        S_te = model.reconstruct(S_test)
        dt = time.perf_counter() - t0
        sm = model.summary()
        print(
            f"     family={family!r:11s}  ortho={ortho!r:10s}  "
            f"train={relative_error(S_train, S_tr):.3e}  "
            f"test={relative_error(S_test, S_te):.3e}  "
            f"[{dt:.2f}s]  "
            f"selected_NL={sm['selected_nonlinear']}/{sm['total_poly_features'] - sm['linear_features']}"
        )

    print("\nDone.")

#%%
if __name__ == "__main__":
    main(r=10, s=100, p=3)
