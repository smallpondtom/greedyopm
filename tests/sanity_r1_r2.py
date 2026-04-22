"""At r=1, GreedyOPM and GreedyFixedOPM must give the SAME reconstruction error.
At r=2 with any modes, they may differ due to incremental-vs-full basis."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

from experiments.data.transport import generate_advecting_pulse
from greedyopm import GreedyFixedOPM, GreedyOPM


def relerr(S, Sa):
    return float(jnp.linalg.norm(S - Sa, "fro") / jnp.linalg.norm(S, "fro"))


if __name__ == "__main__":
    S, _, _ = generate_advecting_pulse(
        pulse_width=5.0e-4, pulse_shift=0.1, speed=5.0,
        final_time=0.20, n_time_samples=400, n_space_samples=2 ** 8,
    )
    S_train = S[:, ::2]

    for r in [1, 2]:
        print(f"\n=== r = {r} ===")
        mF = GreedyFixedOPM(
            r=r, s=30, p=2, gamma=1e-14, n_reorth=1,
            family="monomial", ortho_method="arnoldi", center=True,
        ).fit(S_train)
        eF = relerr(S_train, mF.reconstruct(S_train))

        mS = GreedyOPM(
            r=r, s=30, p=2, gamma_sp=1e-20, gamma=1e-14, n_reorth=1,
            family="monomial", basis="hierarchical",
            ortho_method="arnoldi", center=True,
        ).fit(S_train)
        eS = relerr(S_train, mS.reconstruct(S_train))

        print(f"  Fixed modes  = {list(mF.selected_modes)}, err = {eF:.4e}")
        print(f"  Sparse modes = {list(mS.selected_modes)}, err = {eS:.4e}")
        print(f"  n_linear_ (sparse) = {mS.n_linear_}  "
              f"m_nl_ = {mS.m_nl_}  "
              f"Psi_.shape = {mS.Psi_.shape}")

        # Check Fact 3.3 (ii): (1/n) P_nl @ S_hat^T = 0
        P_nl = mS.Psi_[mS.n_linear_:]
        n = S_train.shape[1]
        fact33_err = float(jnp.linalg.norm(
            (1.0 / n) * P_nl @ mS.S_hat_.T
        ))
        print(f"  Fact 3.3(ii) violation |(1/n) P_nl S_hat^T|_F = {fact33_err:.3e}")