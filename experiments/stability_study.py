"""Stability study for the Greedy-OPM (Jacobian regularization + certificate).

Uses the periodic advection-diffusion transport problem with the TEMPORAL
train / val / test split (test = far future, genuinely out-of-distribution)
and the full nonlinear feature set (gamma_sp = 0), so the effect of the
stability weight mu is isolated.  Three assessments:

1. ``mu-sweep``   -- train / val / test error and the a posteriori Lipschitz
   certificate L_hat(delta) as functions of mu, for BOTH the diagonal
   surrogate shrinkage (``set_mu``) and the exact coupled Jacobian penalty
   (``polish_exact``).  Shows the OOD blow-up at mu = 0, its suppression as
   mu grows, and the monotone decrease of the certificate toward 1.

2. ``certificate`` -- validity check: the certificate upper-bounds the
   pointwise Jacobian norms ||J_Gamma(s_hat)||_2 on the (dilated) probe cloud
   and on the encoded TEST trajectory.

3. ``budget``     -- budget enforcement: for prescribed budgets L, the outer
   mu-loop (``enforce_budget``) returns a decoder whose certificate is <= L;
   reports the resulting mu, certificate, and errors.

Outputs: JSON + plots in experiments/plots/.

Usage:  python experiments/stability_study.py [mu|certificate|budget|all]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "transport"))
sys.path.insert(0, str(_HERE.parent / "src"))

from transport_compare import generate                 # noqa: E402
from exp_split import split_indices                    # noqa: E402
from greedyopm import GreedyOPM                        # noqa: E402

PLOTS = _HERE / "plots"
OUT_JSON = PLOTS / "stability_study.json"

R, S_POOL, DEGREES = 6, 100, [2, 3]
DELTA = 0.1          # extrapolation margin for the certificate probes
GAMMA = 1e-13


def _data():
    S, x, t = generate()
    itr, iva, ite = split_indices(S.shape[1])
    return tuple(jnp.asarray(S[:, idx]) for idx in (itr, iva, ite))


def _fit(S_tr):
    return GreedyOPM(r=R, s=S_POOL, p=DEGREES, gamma_sp=0.0, gamma=GAMMA,
                     ortho_method="arnoldi", center=True, n_reorth=2,
                     grad_dilation=DELTA).fit(S_tr)


def _mu_values(m, n_pts=13):
    m._ensure_dirichlet()
    E = np.asarray(m.dirichlet_nl_)
    E = E[E > 0.0]
    n = float(m.Psi_.shape[1])
    lo = 1e-3 * n / float(np.quantile(E, 0.95))
    hi = 1e6 * n / float(np.quantile(E, 0.05))
    return np.concatenate([[0.0], np.logspace(np.log10(lo), np.log10(hi),
                                              n_pts)])


def mu_sweep(m, S_tr, S_va, S_te):
    """Errors, certificate, and true max Jacobian norm vs mu.

    Compares the diagonal surrogate (whose per-column shrinkage targets the
    certificate directly, driving it monotonically to 1) with the exact
    coupled penalty (which minimizes the TRUE empirical Jacobian norm --
    tighter actual stability, but the triangle-inequality certificate cannot
    see the cancellations it exploits and stays loose).
    """
    pts_dil = (1.0 + DELTA) * m.S_hat_
    rows = []
    for variant in ("surrogate", "exact"):
        for mu in _mu_values(m):
            if variant == "exact":
                m.set_mu(0.0)
                m.set_sparsity(0.0)
                if mu > 0.0:
                    m.polish_exact(float(mu))
            else:
                m.set_mu(float(mu))
                m.set_sparsity(0.0)
            maxJ = float(np.max(np.asarray(
                m.decoder_jacobian_norms(pts_dil))))
            rows.append(dict(
                variant=variant, mu=float(mu),
                train=m.reconstruction_error(S_tr),
                val=m.reconstruction_error(S_va),
                test=m.reconstruction_error(S_te),
                cert=m.lipschitz_certificate(dilation=DELTA),
                maxJ=maxJ,
            ))
            r = rows[-1]
            print(f"  [{variant:9s}] mu={mu:9.2e} train={r['train']:.3e} "
                  f"test={r['test']:.3e} L_hat={r['cert']:.3e} "
                  f"max||J||={maxJ:.3e}")
    return rows


def certificate_check(m, S_te, mu_exact):
    """Certificate must upper-bound the pointwise Jacobian norms."""
    m.set_mu(0.0); m.set_sparsity(0.0); m.polish_exact(mu_exact)
    cert = m.lipschitz_certificate(dilation=DELTA)
    pts_dil = (1.0 + DELTA) * m.S_hat_
    pts_test = m.encode(S_te)
    J_dil = np.asarray(m.decoder_jacobian_norms(pts_dil))
    J_test = np.asarray(m.decoder_jacobian_norms(pts_test))
    out = dict(mu=float(mu_exact), cert=float(cert),
               max_J_dilated=float(J_dil.max()),
               max_J_test=float(J_test.max()),
               holds_dilated=bool(J_dil.max() <= cert * (1 + 1e-9)),
               holds_test=bool(J_test.max() <= cert * (1 + 1e-9)))
    print(f"  certificate L_hat={cert:.4g}; max||J|| dilated cloud="
          f"{out['max_J_dilated']:.4g} (bounded: {out['holds_dilated']}), "
          f"encoded test={out['max_J_test']:.4g} "
          f"(bounded: {out['holds_test']})")
    return out


def budget_sweep(m, S_tr, S_te):
    """enforce_budget drives the certificate below any prescribed L > 1."""
    rows = []
    cert0 = None
    for L in (1e6, 1e4, 1e2, 3e1, 1e1, 3e0):
        m.set_mu(0.0)
        m.set_sparsity(0.0)
        cert0 = m.lipschitz_certificate(dilation=DELTA) if cert0 is None else cert0
        cert = m.enforce_budget(L, dilation=DELTA, mu0=1e-12, growth=10.0)
        rows.append(dict(budget=float(L), mu=float(m.mu), cert=float(cert),
                         train=m.reconstruction_error(S_tr),
                         test=m.reconstruction_error(S_te),
                         satisfied=bool(cert <= L)))
        r = rows[-1]
        print(f"  L={L:9.2e} -> mu={r['mu']:9.2e} L_hat={r['cert']:9.3e} "
              f"train={r['train']:.3e} test={r['test']:.3e} "
              f"ok={r['satisfied']}")
    print(f"  (unregularized certificate: {cert0:.3e})")
    return rows


def plot_mu_sweep(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for variant, color in (("surrogate", "#1f77b4"), ("exact", "#d62728")):
        rs = [r for r in rows if r["variant"] == variant and r["mu"] > 0]
        mus = [r["mu"] for r in rs]
        axes[0].loglog(mus, [r["test"] for r in rs], "-o", color=color,
                       label=f"test ({variant})")
        axes[0].loglog(mus, [r["train"] for r in rs], "--s", color=color,
                       alpha=0.6, label=f"train ({variant})")
        axes[1].loglog(mus, [r["cert"] for r in rs], "-o", color=color,
                       label=f"certificate ({variant})")
        axes[1].loglog(mus, [r["maxJ"] for r in rs], "--s", color=color,
                       alpha=0.6, label=r"max $\|J_\Gamma\|_2$" + f" ({variant})")
    mu0 = [r for r in rows if r["variant"] == "surrogate" and r["mu"] == 0.0]
    if mu0:
        axes[0].axhline(mu0[0]["test"], color="gray", ls=":",
                        label="test (mu=0)")
        axes[1].axhline(mu0[0]["cert"], color="gray", ls=":",
                        label="mu=0")
    axes[0].set_xlabel(r"stability weight $\mu$")
    axes[0].set_ylabel("relative reconstruction error")
    axes[0].set_title("OOD error vs stability weight")
    axes[1].set_xlabel(r"stability weight $\mu$")
    axes[1].set_ylabel(r"$\widehat L_\Gamma(\delta)$ and max $\|J_\Gamma\|_2$")
    axes[1].set_title("Certificate and true Jacobian norm vs stability weight")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Advecting pulse — stability regularization "
                 f"(r={R}, p={DEGREES}, all features active, "
                 r"$\delta=$" + f"{DELTA})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"stability_mu_sweep.{ext}", dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {PLOTS/'stability_mu_sweep.png'}/.pdf")


def plot_budget(rows):
    fig, ax1 = plt.subplots(figsize=(7, 4.6))
    L = [r["budget"] for r in rows]
    ax1.loglog(L, [r["cert"] for r in rows], "-o", color="#d62728",
               label=r"certificate $\widehat L_\Gamma$")
    ax1.loglog(L, L, ":", color="gray", label="budget $L$ (identity)")
    ax1.set_xlabel("prescribed budget $L$")
    ax1.set_ylabel(r"certificate $\widehat L_\Gamma(\delta)$", color="#d62728")
    ax2 = ax1.twinx()
    ax2.loglog(L, [r["test"] for r in rows], "-s", color="#1f77b4",
               label="test error")
    ax2.set_ylabel("relative test error", color="#1f77b4")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_title("Budget enforcement: "
                  r"$\widehat L_\Gamma(\delta) \leq L$ via the outer $\mu$-loop")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"stability_budget.{ext}", dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {PLOTS/'stability_budget.png'}/.pdf")


def main(phase="all"):
    PLOTS.mkdir(exist_ok=True)
    S_tr, S_va, S_te = _data()
    print(f"[stability] transport d={S_tr.shape[0]} "
          f"train {S_tr.shape[1]} / val {S_va.shape[1]} / test {S_te.shape[1]}")
    m = _fit(S_tr)
    out = json.load(open(OUT_JSON)) if OUT_JSON.exists() else {}

    if phase in ("all", "mu"):
        print("[stability] mu sweep (surrogate vs exact):")
        out["mu_sweep"] = mu_sweep(m, S_tr, S_va, S_te)
        plot_mu_sweep(out["mu_sweep"])

    if phase in ("all", "certificate"):
        print("[stability] certificate validity:")
        rows = out.get("mu_sweep", [])
        ex = [r for r in rows if r["variant"] == "exact" and r["mu"] > 0]
        mu_star = (min(ex, key=lambda r: r["val"])["mu"] if ex else 1e2)
        out["certificate"] = certificate_check(m, S_te, mu_star)

    if phase in ("all", "budget"):
        print("[stability] budget enforcement:")
        out["budget"] = budget_sweep(m, S_tr, S_te)
        plot_budget(out["budget"])

    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print(f"[stability] saved {OUT_JSON}")


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[stability] elapsed {time.time()-t0:.1f}s")
