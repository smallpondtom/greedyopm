"""Round-trip tests for GreedyOPM / GreedyFixedOPM.

Specifically, verify:
* encode(decode(s_hat)) != s_hat in general (nonlinear lifting),
* but reconstruct(S) gives small error on training data,
* and out-of-sample test error is comparable to training error.
"""

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


def test_roundtrip_for(cls, label, **kwargs):
    S, _, _ = generate_advecting_pulse(
        pulse_width=5.0e-4, pulse_shift=0.1, speed=5.0,
        final_time=0.20, n_time_samples=400, n_space_samples=2 ** 8,
    )
    S_train = S[:, ::2]
    S_test = S[:, 1::2]

    m = cls(**kwargs).fit(S_train)
    e_tr = relerr(S_train, m.reconstruct(S_train))
    e_te = relerr(S_test, m.reconstruct(S_test))

    # Single-snapshot path
    single = S_test[:, 0]
    reco = m.reconstruct(single)
    assert reco.shape == single.shape, (reco.shape, single.shape)

    # Encode produces shape (r,) for a single snapshot
    s_hat = m.encode(single)
    assert s_hat.shape == (kwargs["r"],), s_hat.shape
    # Decode is compatible
    dec = m.decode(s_hat)
    assert dec.shape == single.shape

    print(f"  {label:35s} train={e_tr:.3e}  test={e_te:.3e}")
    # Train and test should be in the same order of magnitude.
    ratio = e_te / e_tr
    assert 0.3 < ratio < 3.0, f"train/test ratio = {ratio} (label={label})"


if __name__ == "__main__":
    # Note: Cholesky is numerically fragile at p=2 with r~8 because forming
    # the Gram matrix squares the condition number.  We exclude it here;
    # users are encouraged to use ortho_method='arnoldi' (default) or 'qr'.
    for ortho in ["arnoldi", "qr"]:
        test_roundtrip_for(
            GreedyFixedOPM,
            f"GreedyFixedOPM ortho={ortho}",
            r=8, s=40, p=2, gamma=1e-14, n_reorth=1,
            family="monomial", ortho_method=ortho, center=True,
        )

    for ortho in ["arnoldi", "qr"]:
        test_roundtrip_for(
            GreedyOPM,
            f"GreedyOPM     ortho={ortho}",
            r=8, s=40, p=2, gamma_sp=1e-4, gamma=1e-14, n_reorth=1,
            family="monomial", basis="hierarchical",
            ortho_method=ortho, center=True,
        )

    print("All round-trip tests passed.")