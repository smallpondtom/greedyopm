"""POD singular-value decay of the parametric (angle-of-attack) NACA dataset.

Builds the (area-weighted) velocity snapshot matrix from the developed window
(t >= T_DEV) of every SU2 run, and reports:
  (a) the normalized singular-value spectrum of the combined parametric manifold,
  (b) the relative projection (reconstruction) error vs reduced dimension r,
contrasted with a single angle's own decay, to show that the parametric family
decays much more slowly (the Kolmogorov-n-width point).
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PLOTS = os.path.join(HERE, "plots")
import sys
sys.path.insert(0, HERE)
from problem import T_DEV, load_mesh  # noqa: E402


def load_state(path):
    """Return the developed-window weighted velocity snapshots (2N, k) and alpha."""
    d = np.load(path)
    keep = d["t"] >= T_DEV
    w = load_mesh()["sqrt_w"]
    u = d["u"][keep].astype(np.float64) * w
    v = d["v"][keep].astype(np.float64) * w
    S = np.concatenate([u, v], axis=1).T                    # (2N, k)
    return S, float(d["alpha_deg"])


def pod_error(S):
    """Method-of-snapshots singular values and relative projection error vs r."""
    c = S.mean(axis=1, keepdims=True)
    A = S - c
    G = A.T @ A
    w = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    sv = np.sqrt(w)
    energy = np.cumsum(w) / np.sum(w)
    err = np.sqrt(np.clip(1.0 - energy, 0, None))
    return sv, err


def main():
    files = sorted(glob.glob(os.path.join(DATA, "naca_a*.npz")))
    if not files:
        raise SystemExit("no data/naca_a*.npz found")

    blocks, angles = [], []
    for f in files:
        S, a = load_state(f)
        blocks.append(S)
        angles.append(a)
        print(f"  alpha={a:5.1f}  window snapshots={S.shape[1]}", flush=True)
    S_all = np.concatenate(blocks, axis=1)
    print(f"parametric snapshot matrix: {S_all.shape}", flush=True)

    sv_all, err_all = pod_error(S_all)
    # a single representative angle (middle of the range) for contrast
    mid = len(blocks) // 2
    sv_one, err_one = pod_error(blocks[mid])

    rmax = 60
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    ax = axes[0]
    ax.semilogy(np.arange(1, min(rmax, sv_all.size) + 1),
                sv_all[:rmax] / sv_all[0], "-o", ms=4, color="#1f77b4",
                label=f"parametric (all {len(angles)} angles)")
    ax.semilogy(np.arange(1, min(rmax, sv_one.size) + 1),
                sv_one[:rmax] / sv_one[0], "--s", ms=4, color="#ff7f0e",
                label=rf"single angle ($\alpha={angles[mid]:.0f}^\circ$)")
    ax.set_xlabel("index $i$"); ax.set_ylabel(r"$\sigma_i/\sigma_1$")
    ax.set_title("POD singular-value spectrum")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()

    ax = axes[1]
    r = np.arange(1, rmax + 1)
    ax.semilogy(r, err_all[:rmax], "-o", ms=4, color="#1f77b4",
                label=f"parametric (all {len(angles)} angles)")
    ax.semilogy(r, err_one[:rmax], "--s", ms=4, color="#ff7f0e",
                label=rf"single angle ($\alpha={angles[mid]:.0f}^\circ$)")
    for lvl in (1e-1, 1e-2):
        ax.axhline(lvl, color="grey", ls=":", lw=0.8)
    ax.set_xlabel(r"reduced dimension $r$")
    ax.set_ylabel("relative projection error")
    ax.set_title("POD reconstruction error vs $r$")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()

    fig.suptitle(rf"NACA 0012, $Re=1000$, parametric over angle of attack "
                 rf"$\alpha \in [{min(angles):.0f}, {max(angles):.0f}]^\circ$",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "sv_decay.png"), dpi=170)
    fig.savefig(os.path.join(PLOTS, "sv_decay.pdf"))
    print("saved plots/sv_decay.png", flush=True)

    # print a small table
    print("\n r   parametric_err   single_angle_err")
    for rr in [2, 5, 10, 20, 30, 40, 50]:
        if rr <= err_all.size:
            print(f"{rr:3d}   {err_all[rr-1]:.3e}        {err_one[min(rr-1, err_one.size-1)]:.3e}")


if __name__ == "__main__":
    main()
