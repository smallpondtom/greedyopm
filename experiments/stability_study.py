"""Stability study for the budget-constrained Greedy-OPM.

Transport problem, temporal train/val/test split, quadratic-plus-cubic
dictionary at r=6 with gamma_sp=0 (every feature eligible), so the Lipschitz
budget L is the only control.  Two assessments:

1. ``budget``      -- for each prescribed budget L a fresh budget-aware fit
   is run (in-loop knapsack admission).  Records train/test errors, the
   number of admitted features, the a posteriori certificate
   L_hat_Gamma(delta) <= L (holds BY CONSTRUCTION), and the measured maximal
   pointwise Jacobian norm max_l ||J_Gamma||_2 on the dilated probe cloud.
   Shows the OOD blow-up in the unconstrained limit, its suppression as L
   tightens, and the accuracy/guarantee trade-off.

2. ``certificate`` -- validity check: the certificate must upper-bound the
   pointwise Jacobian norms on both the dilated training cloud and the
   encoded test trajectory.

Outputs plots/stability_budget.{png,pdf} and plots/stability_study.json.

Usage:  python experiments/stability_study.py [budget|certificate|plots|all]
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
L_GRID = [1.5, 3.0, 10.0, 30.0, 1e2, 316.0, 1e3, 1e4, 1e5, 1e6, 1e7, np.inf]


def _data():
    S, x, t = generate()
    itr, iva, ite = split_indices(S.shape[1])
    return tuple(jnp.asarray(S[:, idx]) for idx in (itr, iva, ite))


def _fit(S_tr, L):
    budget = None if np.isinf(L) else float(L)
    return GreedyOPM(r=R, s=S_POOL, p=DEGREES, gamma_sp=0.0, gamma=GAMMA,
                     ortho_method="arnoldi", center=True, n_reorth=2,
                     grad_dilation=DELTA, budget=budget).fit(S_tr)


def _max_jac(m, S_hat):
    return float(jnp.max(m.decoder_jacobian_norms(S_hat)))


def budget_sweep(S_tr, S_va, S_te):
    """Fresh budget-aware fit per L; certificate holds by construction."""
    rows = []
    for L in L_GRID:
        m = _fit(S_tr, L)
        probe = (1.0 + DELTA) * m.S_hat_
        row = dict(
            L=(None if np.isinf(L) else float(L)),
            n_active=int(m.n_selected_nonlinear),
            m_total=int(m.m_nl_),
            cert=float(m.certificate_),
            max_jac_dilated=_max_jac(m, probe),
            train=m.reconstruction_error(S_tr),
            val=m.reconstruction_error(S_va),
            test=m.reconstruction_error(S_te))
        rows.append(row)
        print(f"  L={L:9.3g} act={row['n_active']:3d}/{row['m_total']} "
              f"cert={row['cert']:9.3g} maxJ={row['max_jac_dilated']:9.3g} "
              f"train={row['train']:.3e} test={row['test']:.3e}")
    return rows


def certificate_check(S_tr, S_te):
    """Certificate must upper-bound the pointwise Jacobian norms."""
    m = _fit(S_tr, 1e3)
    probe = (1.0 + DELTA) * m.S_hat_
    Sh_te = m.encode(S_te)
    out = dict(cert=float(m.certificate_),
               max_jac_dilated=_max_jac(m, probe),
               max_jac_test=_max_jac(m, Sh_te),
               n_active=int(m.n_selected_nonlinear))
    ok = (out["max_jac_dilated"] <= out["cert"] * (1 + 1e-9))
    print(f"  cert={out['cert']:.4g}  maxJ(dilated)={out['max_jac_dilated']:.4g} "
          f"maxJ(test)={out['max_jac_test']:.4g}  valid={ok}")
    out["valid_on_dilated"] = bool(ok)
    return out


def plots(res):
    rows = res["budget"]
    L = [1e8 if e["L"] is None else e["L"] for e in rows]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.loglog(L, [e["train"] for e in rows], "--s", color="#9467bd",
              label="train")
    ax.loglog(L, [e["test"] for e in rows], "-o", color="#d62728",
              label="test")
    ax.set_xlabel(r"prescribed budget $L$  ($10^8$ = unconstrained)")
    ax.set_ylabel("relative reconstruction error")
    ax.set_title("error vs budget")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.loglog(L, [e["cert"] for e in rows], "-o", color="#d62728",
              label=r"certificate $\widehat L_\Gamma(\delta)$")
    ax.loglog(L, [e["max_jac_dilated"] for e in rows], "--^",
              color="#1f77b4", label=r"measured $\max_l\|J_\Gamma\|_2$")
    Lf = [x for x in L if x < 1e8]
    ax.loglog(Lf, Lf, ":", color="gray", label="identity ($=L$)")
    ax.set_xlabel(r"prescribed budget $L$")
    ax.set_title("certificate $\\leq L$ by construction")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    ax.semilogx(L, [e["n_active"] for e in rows], "-o", color="#2ca02c")
    ax.axhline(rows[0]["m_total"], color="gray", ls=":", lw=1)
    ax.set_xlabel(r"prescribed budget $L$")
    ax.set_ylabel("admitted nonlinear features")
    ax.set_title("dictionary usage vs budget")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Budget-constrained selection on the transport problem "
                 f"(r={R}, p={DEGREES}, $\\gamma_{{sp}}=0$, "
                 f"$\\delta$={DELTA})", fontsize=14)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"stability_budget.{ext}", dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {PLOTS/'stability_budget.png'}/.pdf")


def main(phase="all"):
    PLOTS.mkdir(exist_ok=True)
    res = json.load(open(OUT_JSON)) if OUT_JSON.exists() else {}
    S_tr, S_va, S_te = _data()
    if phase in ("all", "budget") and "budget" not in res:
        print("[stability] budget sweep:")
        res["budget"] = budget_sweep(S_tr, S_va, S_te)
        json.dump(res, open(OUT_JSON, "w"), indent=1)
    if phase in ("all", "certificate") and "certificate" not in res:
        print("[stability] certificate validity:")
        res["certificate"] = certificate_check(S_tr, S_te)
        json.dump(res, open(OUT_JSON, "w"), indent=1)
    if phase in ("all", "plots"):
        plots(res)


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[stability] elapsed {time.time()-t0:.1f}s")
