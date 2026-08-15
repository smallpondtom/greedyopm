"""Verify that soft-thresholding sparsifies NL features on transport."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

from experiments.data.transport import generate_advecting_pulse
from greedyopm import GreedyOPM


def relerr(S, Sa):
    return float(jnp.linalg.norm(S - Sa, "fro") / jnp.linalg.norm(S, "fro"))


if __name__ == "__main__":
    S, _, _ = generate_advecting_pulse(
        pulse_width=5.0e-4, pulse_shift=0.1, speed=5.0,
        final_time=0.20, n_time_samples=800, n_space_samples=2 ** 9,
    )
    S_train = S[:, ::2]
    S_test = S[:, 1::2]

    r, s, p = 10, 60, 2
    print(f"r={r}, s={s}, p={p}, train={S_train.shape}")
    print()
    print(f"{'gamma_sp':>10s}  {'train err':>12s}  {'test err':>12s}  "
          f"{'NL active':>12s}  {'modes':>40s}")
    print("-" * 100)

    for gamma_sp in [0.0, 1e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]:
        m = GreedyOPM(
            r=r, s=s, p=p,
            gamma_sp=gamma_sp,
            gamma=1e-14,
            n_reorth=1,
            family="monomial",
            basis="hierarchical",
            ortho_method="arnoldi",
            center=True,
        ).fit(S_train)
        et = relerr(S_train, m.reconstruct(S_train))
        ete = relerr(S_test, m.reconstruct(S_test))
        sm = m.summary()
        modes_str = str(list(int(j) for j in m.selected_modes))[:38]
        print(f"{gamma_sp:10.2e}  {et:12.4e}  {ete:12.4e}  "
              f"{sm['selected_nonlinear']:5d} / {sm['total_poly_features'] - sm['linear_features']:<4d}  "
              f"{modes_str:>40s}")