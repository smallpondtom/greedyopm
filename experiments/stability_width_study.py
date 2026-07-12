"""Numerical studies for the stability + width-matching note.

Two studies (cf. Section 4 of ``stability_widths_note.tex``):

1. ``ood_sweep``   -- stability regularization sweep: train on an initial
   time window, test beyond it; report train/test error and the Lipschitz
   certificate ``L_hat`` as a function of ``mu``.
2. ``width_check`` -- empirical width matching: compare the greedy-OPM
   mean-squared error ``J/n`` against the lower bound
   ``d_{t(r,p)-1}^{(2)}(S)^2 = (1/n) sum_{i > t-1} sigma_i^2``
   (Proposition `empirical BGH lower bound`), and report the effective
   matching constant ``sqrt(1 + kappa^2)`` from Theorem `two-sided bound`.

Usage
-----
    pixi run python experiments/stability_width_study.py [--quick]
"""

from __future__ import annotations

import argparse
from math import comb

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from greedyopm import GreedyOPM


# ---------------------------------------------------------------------------
# Data: traveling Gaussian bump (transport-dominated, slow n-width decay)
# ---------------------------------------------------------------------------


def traveling_wave(n_t: int, t_max: float, d: int = 240, width: float = 0.08):
    x = np.linspace(-1.0, 1.0, d)
    t = np.linspace(0.0, t_max, n_t)
    S = np.exp(-((x[:, None] - (-0.6 + t[None, :])) ** 2) / (2 * width**2))
    return jnp.asarray(S)


# ---------------------------------------------------------------------------
# Study 1: OOD sweep over mu
# ---------------------------------------------------------------------------


def ood_sweep(r=4, s=16, p=3, delta=0.25, quick=False):
    n_t = 120 if quick else 240
    S_all = traveling_wave(n_t=n_t, t_max=0.9)
    n_train = int(0.6 * n_t)
    S_tr, S_te = S_all[:, :n_train], S_all[:, n_train:]
    n = S_tr.shape[1]

    # natural mu scale from the unregularized Dirichlet energies
    probe = GreedyOPM(r=r, s=s, p=p, mu=1e-300, grad_dilation=delta).fit(S_tr)
    E = np.asarray(probe.dirichlet_nl_)
    mu_med = n / max(float(np.median(E)), 1e-300)

    print(f"\n=== OOD sweep (r={r}, s={s}, p={p}, delta={delta}) ===")
    print(f"Dirichlet energies: median={np.median(E):.3g}, max={E.max():.3g}")
    print(f"{'mu':>12} {'train':>10} {'test(OOD)':>10} {'L_hat':>12} "
          f"{'L_hat(d=0)':>12} {'#active':>8}")
    for fac in [0.0, 1e-2, 1e-1, 1.0, 1e1, 1e2]:
        mu = fac * mu_med
        m = GreedyOPM(r=r, s=s, p=p, mu=mu, grad_dilation=delta).fit(S_tr)
        print(f"{mu:12.3e} {m.reconstruction_error(S_tr):10.4f} "
              f"{m.reconstruction_error(S_te):10.4f} "
              f"{m.lipschitz_certificate(dilation=delta):12.3e} "
              f"{m.lipschitz_certificate(dilation=0.0):12.3e} "
              f"{m.n_selected_nonlinear:8d}")


# ---------------------------------------------------------------------------
# Study 2: width matching
# ---------------------------------------------------------------------------


def width_check(p=2, quick=False):
    n_t = 150 if quick else 400
    S = traveling_wave(n_t=n_t, t_max=0.9)
    S_tilde = S - jnp.mean(S, axis=1, keepdims=True)
    sigma = np.asarray(jnp.linalg.svd(S_tilde, compute_uv=False))
    n = S.shape[1]
    tail = np.concatenate([np.cumsum(sigma[::-1] ** 2)[::-1], [0.0]])

    print(f"\n=== Width matching (p={p}) ===")
    print(f"{'r':>3} {'t-1':>5} {'s':>5} {'sqrt(J/n)':>12} "
          f"{'d_(t-1)':>12} {'ratio':>8}")
    for r in [2, 3, 4, 5, 6]:
        t_np = comb(r + p, p)              # 1 + r + m_bar
        s_pool = min(t_np - 1, len(sigma), n - 1)
        if s_pool <= r:
            continue
        m = GreedyOPM(r=r, s=s_pool, p=p, gamma_sp=0.0, gamma=1e-10).fit(S)
        err = m.reconstruction_error(S, relative=False) / np.sqrt(n)
        lb_idx = min(t_np - 1, len(sigma))
        lower = np.sqrt(tail[lb_idx] / n)
        ratio = err / max(lower, 1e-300)
        print(f"{r:3d} {t_np-1:5d} {s_pool:5d} {err:12.4e} "
              f"{lower:12.4e} {ratio:8.3f}")
    print("ratio = effective sqrt(1 + kappa^2) of Corollary `matching`;\n"
          "ratio >= 1 always (lower bound), close to 1 means the BGH bound "
          "is attained.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ood_sweep(quick=args.quick)
    width_check(p=2, quick=args.quick)
    width_check(p=3, quick=args.quick)
