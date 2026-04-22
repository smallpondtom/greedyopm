"""
Compare reconstruction error vs reduced dimension r for:
  - POD (linear baseline)
  - Greedy-QM (Schwerdtner & Peherstorfer, quadratic monomials)
  - Greedy-CM (Schwerdtner & Peherstorfer, cubic monomials)
  - Greedy-Fixed-OPM p=[2]   (orthogonal quadratic)
  - Greedy-Fixed-OPM p=[3]   (orthogonal cubic only)
  - Greedy-Fixed-OPM p=[2,3] (orthogonal quadratic + cubic)
on the advecting Gaussian pulse problem.
"""

#%%
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from greedyopm import GreedyFixedOPM
from qm.quadmani import (
    quadmani_greedy,
    quadratic_mapping,
    cubic_mapping,
    shift_data,
    linear_reduce,
    lift_quadratic,
)


#%% ===========================================================================
# Data generation — advecting Gaussian pulse
# =============================================================================

def generate_advecting_pulse(
    pulse_width=5.0e-4,
    pulse_shift=0.1,
    speed=5.0,
    final_time=0.20,
    n_time_samples=2000,
    n_space_samples=1024,
):
    """Advecting Gaussian pulse  s(t, x) = s_0(x − c·t).

    Following Schwerdtner & Peherstorfer (2024), Eq. (18):
        s_0(x) = (1/sqrt(w·π)) exp(-(x - μ)^2 / w),   x ∈ [0, 1].
    """
    xspan = np.linspace(0, 1, n_space_samples)
    tspan = np.linspace(0, final_time, n_time_samples)
    S = np.zeros((n_space_samples, n_time_samples))
    for j, t in enumerate(tspan):
        center = pulse_shift + speed * t
        S[:, j] = (
            np.exp(-((xspan - center) ** 2) / pulse_width)
            / np.sqrt(pulse_width * np.pi)
        )
    return S, xspan, tspan


#%% ===========================================================================
# Helpers
# =============================================================================

def relative_error(S_true, S_approx):
    return float(
        jnp.linalg.norm(S_true - S_approx, "fro")
        / jnp.linalg.norm(S_true, "fro")
    )


def pod_errors_at_r(S_train, S_test, r, s_ref):
    """Compute POD (leading-r) train/test reconstruction error."""
    U, _, _ = jnp.linalg.svd(S_train - s_ref, full_matrices=False)
    U_r = U[:, :r]
    proj = lambda S: s_ref + U_r @ (U_r.T @ (S - s_ref))
    return relative_error(S_train, proj(S_train)), relative_error(S_test, proj(S_test))


def qm_errors_at_r(S_train, S_test, r, s_pool, gamma, feature_map):
    """Run Greedy-QM/CM at dimension r and return train/test errors."""
    V, Vbar, mu, idx = quadmani_greedy(
        S_train, r=r, n_vectors_to_check=s_pool,
        reg_magnitude=gamma,
        idx_in=np.array([], dtype=int),
        feature_map=feature_map,
        shifting=True,
    )
    mu = mu.reshape(-1, 1)
    def recon(S):
        Sh = V.T @ (S - mu)
        return mu + V @ Sh + Vbar @ feature_map(Sh)
    return relative_error(S_train, recon(S_train)), relative_error(S_test, recon(S_test))


def opm_errors_at_r(S_train, S_test, r, s_pool, p, gamma, n_reorth, 
                    R_rank_tol, family, ortho_method):
    """Run Greedy-Fixed-OPM at dimension r and return train/test errors."""
    model = GreedyFixedOPM(
        r=r, s=s_pool, p=p, gamma=gamma,
        n_reorth=n_reorth, R_rank_tol=R_rank_tol,
        family=family, ortho_method=ortho_method, center=True,
    )
    model.fit(S_train)
    err_train = model.reconstruction_error(S_train, relative=True)
    err_test = model.reconstruction_error(S_test, relative=True)
    return err_train, err_test


# =============================================================================
# Main experiment
# =============================================================================

#%%
if __name__ == "__main__":

    # ---- Parameters ---------------------------------------------------------
    r_max = 10
    s_pool = 100
    gamma_qm = 1e-15  
    gamma_cm = 1e-12
    gamma_opm = 1e-15 

    #%% ---- Generate data ----------------------------------------------------
    print("[1/2] Generating advecting pulse data ...")
    S_np, xspan, tspan = generate_advecting_pulse(
        pulse_width=5.0e-4,
        pulse_shift=0.1,
        speed=5.0,
        final_time=0.2,
        n_time_samples=1000,
        n_space_samples=512,
    )
    S = jnp.array(S_np)
    d, k = S.shape
    print(f"    Snapshot matrix: {d} x {k}")

    # Train / test split (even / odd columns)
    S_train = S[:, ::2]
    S_test = S[:, 1::2]
    k_train, k_test = S_train.shape[1], S_test.shape[1]
    print(f"    Train: {k_train},  Test: {k_test}")

    s_ref = jnp.mean(S_train, axis=1, keepdims=True)

    #%% ---- Sweep over r -----------------------------------------------------
    print(f"\n[2/2] Sweeping r = 1 … {r_max}  (s = {s_pool})")
    rs = list(range(1, r_max + 1))

    results = {
        "POD":                  {"train": [], "test": []},
        "Greedy-QM":            {"train": [], "test": []},
        "Greedy-CM":            {"train": [], "test": []},
        "Greedy-OPM $p$=[2]":   {"train": [], "test": []},
        # "Greedy-OPM $p$=[3]":   {"train": [], "test": []},
        "Greedy-OPM $p$=[2,3]": {"train": [], "test": []},
    }
    timings = {k: [] for k in results}

    for r in rs:
        print(f"\n  r = {r}")

        # --- POD ---
        t0 = time.time()
        tr, te = pod_errors_at_r(S_train, S_test, r, s_ref)
        timings["POD"].append(time.time() - t0)
        results["POD"]["train"].append(tr)
        results["POD"]["test"].append(te)
        print(f"    POD                  train={tr:.4e}  test={te:.4e}")

        # --- Greedy-QM ---
        t0 = time.time()
        tr, te = qm_errors_at_r(
            S_train, S_test, r, s_pool, gamma_qm, quadratic_mapping,
        )
        timings["Greedy-QM"].append(time.time() - t0)
        results["Greedy-QM"]["train"].append(tr)
        results["Greedy-QM"]["test"].append(te)
        print(f"    Greedy-QM            train={tr:.4e}  test={te:.4e}")

        # --- Greedy-CM ---
        t0 = time.time()
        tr, te = qm_errors_at_r(
            S_train, S_test, r, s_pool, gamma_cm, cubic_mapping,
        )
        timings["Greedy-CM"].append(time.time() - t0)
        results["Greedy-CM"]["train"].append(tr)
        results["Greedy-CM"]["test"].append(te)
        print(f"    Greedy-CM            train={tr:.4e}  test={te:.4e}")

        # --- OPM p=[2] ---
        t0 = time.time()
        tr, te = opm_errors_at_r(
            S_train, S_test, r, s_pool, [2], gamma_opm, 
            2, 1e-8, "monomial", "arnoldi",
        )
        timings["Greedy-OPM $p$=[2]"].append(time.time() - t0)
        results["Greedy-OPM $p$=[2]"]["train"].append(tr)
        results["Greedy-OPM $p$=[2]"]["test"].append(te)
        print(f"    Greedy-OPM p=[2]     train={tr:.4e}  test={te:.4e}")

        # # --- OPM p=[3] ---
        # t0 = time.time()
        # tr, te = opm_errors_at_r(
        #     S_train, S_test, r, s_pool, [3], gamma_opm, 
        #     2, 1e-4, "monomial", "arnoldi",
        # )
        # timings["Greedy-OPM $p$=[3]"].append(time.time() - t0)
        # results["Greedy-OPM $p$=[3]"]["train"].append(tr)
        # results["Greedy-OPM $p$=[3]"]["test"].append(te)
        # print(f"    Greedy-OPM p=[3]     train={tr:.4e}  test={te:.4e}")

        # --- OPM p=[2,3] ---
        t0 = time.time()
        tr, te = opm_errors_at_r(
            S_train, S_test, r, s_pool, [2, 3], gamma_opm, 
            2, 1e-4, "monomial", "arnoldi",
        )
        timings["Greedy-OPM $p$=[2,3]"].append(time.time() - t0)
        results["Greedy-OPM $p$=[2,3]"]["train"].append(tr)
        results["Greedy-OPM $p$=[2,3]"]["test"].append(te)
        print(f"    Greedy-OPM p=[2,3]   train={tr:.4e}  test={te:.4e}")

    #%% ---- Plotting --------------------------------------------------------
    print("\nGenerating plots ...")

    # Style matching pulse.py
    method_styles = {
        "POD":                  dict(fmt="-o",   color="black",      lw=4, ms=10),
        "Greedy-QM":            dict(fmt="--^",  color="#1f77b4",  lw=3, ms=10),
        "Greedy-CM":            dict(fmt="--v",  color="#ff7f0e",  lw=3, ms=10),
        "Greedy-OPM $p$=[2]":   dict(fmt="-s",   color="#2ca02c",  lw=3, ms=10),
        # "Greedy-OPM $p$=[3]":   dict(fmt="-D",   color="#d62728",  lw=3, ms=10),
        "Greedy-OPM $p$=[2,3]": dict(fmt="-P",   color="#9467bd",  lw=3, ms=10),
    }

    for split, title_suffix in [("train", "Training Data"), ("test", "Test Data")]:
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))

        for name, style in method_styles.items():
            errs = results[name][split]
            ax.semilogy(
                rs, errs, style["fmt"],
                label=name,
                color=style["color"],
                linewidth=style["lw"],
                markersize=style["ms"],
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(r"Reduced Dimension $r$", fontsize=22)
        ax.set_ylabel("Relative Reconstruction Error", fontsize=22)
        ax.set_title(
            f"Advecting Pulse — {title_suffix}", fontsize=24, pad=12,
        )
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_xticks(rs)
        ax.set_xlim(0.5, r_max + 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(
            fontsize=16, loc="center left",
            bbox_to_anchor=(1.02, 0.5), frameon=False,
        )
        plt.tight_layout()
        fname = f"plots/transport_error_vs_r_{split}.pdf"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        fname = f"plots/transport_error_vs_r_{split}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"    Saved {fname}")
        plt.close(fig)

    # ---- Combined train + test (side by side) ----
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=True)

    for ax, split, title_suffix in zip(
        axes, ["train", "test"], ["Training Data", "Test Data"]
    ):
        for name, style in method_styles.items():
            errs = results[name][split]
            ax.semilogy(
                rs, errs, style["fmt"],
                label=name,
                color=style["color"],
                linewidth=style["lw"],
                markersize=style["ms"],
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(r"Reduced Dimension $r$", fontsize=22)
        ax.set_title(title_suffix, fontsize=24, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_xticks(rs)
        ax.set_xlim(0.5, r_max + 0.5)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Relative Reconstruction Error", fontsize=22)

    # Single legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=18,
        bbox_to_anchor=(0.5, -0.10), frameon=False, handlelength=3,
    )
    fig.suptitle(
        "Advecting Pulse — Reconstruction Error vs Reduced Dimension",
        fontsize=26, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        "plots/transport_error_vs_r_combined.pdf",
        dpi=300, bbox_inches="tight",
    )
    print("    Saved transport_error_vs_r_combined.pdf")
    plt.savefig(
        "plots/transport_error_vs_r_combined.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ---- Print timing summary ----
    print("\nTiming summary (total wall-clock seconds over all r):")
    for name, ts in timings.items():
        print(f"  {name:20s}  {sum(ts):.1f}s")

    print("\nDone.")
# %%
