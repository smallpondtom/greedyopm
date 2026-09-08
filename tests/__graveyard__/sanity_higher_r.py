"""Sanity at r >= 3 with the full-rebuild greedy: verify Fixed and Sparse
pick the SAME modes when gamma_sp → 0, and their reconstruction errors match."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

from experiments.__old__.data.transport import generate_advecting_pulse
from greedyopm import GreedyFixedOPM, GreedyOPM


def relerr(S, Sa):
    return float(jnp.linalg.norm(S - Sa, "fro") / jnp.linalg.norm(S, "fro"))


if __name__ == "__main__":
    S, _, _ = generate_advecting_pulse(
        pulse_width=5.0e-4, pulse_shift=0.1, speed=5.0,
        final_time=0.20, n_time_samples=400, n_space_samples=256,
    )
    S_train = S[:, ::2]

    for r in [3, 4, 5, 6, 8]:
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

        print(f"\n=== r = {r} ===")
        print(f"  Fixed modes  = {list(mF.selected_modes)}, err = {eF:.4e}")
        print(f"  Sparse modes = {list(mS.selected_modes)}, err = {eS:.4e}")
        same_modes = list(mF.selected_modes) == list(mS.selected_modes)
        err_match = abs(eF - eS) / max(eF, eS) < 1e-10
        print(f"  same modes? {same_modes}   err match? {err_match}")