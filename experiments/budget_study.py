"""Lipschitz-budget study for Greedy-OPM (pure Algorithm 4 pipeline).

For each experiment, at its featured reduced dimension, sweep the prescribed
budget L and record how the reconstruction errors (train / validation /
test), the number of selected features, and the enforced stability weight mu
change with L.  For every L the remaining hyperparameters (gamma, gamma_sp,
mode-selection variant) are re-tuned on the validation window with mu set to
the smallest value satisfying the budget (``exp_methods._grid_curves_budget``:
the fixed point of the outer mu-loop, surrogate lifting throughout, no exact
polish).

The per-experiment budget L* used by the main comparison sweeps is the
smallest L whose tuned validation error is within ``L_TOL`` of the best over
all L (most stability at no validation cost).

Usage:  python budget_study.py [transport|swe|nse|plots|all]
Outputs: plots/budget_study.json, plots/budget_study.{png,pdf}
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

from exp_split import split_indices                    # noqa: E402
from exp_methods import (fit_opm_tuned,                # noqa: E402
                         _fit_opm_base, _grid_curves, _select_config,
                         _select_variant, _apply_opm_params)

PLOTS = _HERE / "plots"
OUT_JSON = PLOTS / "budget_study.json"

L_GRID = [1.5, 3.0, 10.0, 30.0, 100.0, 316.0, 1e3, 1e4, 1e5, 1e6, np.inf]
L_TOL = 0.02
WALL_BUDGET = float(__import__('os').environ.get('BS_BUDGET', '32'))          # smallest L whose val error is within 2% of the best
DEGREE_SETS = {"p2": [2], "p23": [2, 3]}

# featured reduced dimension and tuning grids per experiment
GAMMA_FACTORS = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0)
EXPS = {
    "transport": dict(r=8, tol=0.02, variant_tol=0.02),
    "swe": dict(r=8, tol=0.005, variant_tol=0.08),
    "nse": dict(r=10, tol=0.02, variant_tol=0.02),
}


def _load(exp):
    if exp == "transport":
        from transport_compare import generate, CFG
        S, x, t = generate()
        itr, iva, ite = split_indices(S.shape[1])
        blocks = (S[:, itr], S[:, iva], S[:, ite])
    elif exp == "swe":
        S = np.load(_HERE / "shallow_water" / "swe_height_cache.npz")["S_h"]
        sys.path.insert(0, str(_HERE / "shallow_water"))
        sys.path.insert(0, str(_HERE / "data"))
        from shallow_water_compare import CFG
        itr, iva, ite = split_indices(S.shape[1])
        blocks = (S[:, itr], S[:, iva], S[:, ite])
    else:
        D = np.load(_HERE / "nse" / "data" / "nse_compressed.npz")
        sys.path.insert(0, str(_HERE / "nse"))
        from nse_compare import CFG
        blocks = (D["A_train"], D["A_val"], D["A_test"])
    cfg = dict(CFG)
    cfg["gamma_factors"] = GAMMA_FACTORS       # ridge grid for every problem
    cfg.pop("budget", None)                    # the study drives L itself
    return tuple(jnp.asarray(b) for b in blocks), cfg


def study_one(exp, deg, L, blocks, cfg, out, key):
    """One (experiment, degree-set, budget) cell: full budget-aware fit+tune.

    The in-loop greedy mode selection runs under the budget L (knapsack
    admission per candidate), and (gamma, gamma_sp, variant) are tuned on
    the validation window -- identical to the main comparison protocol.
    STAGED per variant (checkpointed under ``__stage:`` keys) so each
    sandbox call fits at most one greedy variant; returns None while staging.
    """
    (S_tr, S_va, S_te) = blocks
    p = EXPS[exp]
    r = p["r"]
    cfg = dict(cfg)
    cfg["budget"] = None if np.isinf(L) else float(L)
    stages = {}
    for sel, sp in (("greedy", None), ("leading", r)):
        skey = f"__stage:{key}:{sel}"
        if skey not in out:
            m = _fit_opm_base(S_tr, r, DEGREE_SETS[deg], cfg, s_pool=sp)
            curves = _grid_curves(m, S_tr, S_va, cfg,
                                  cfg.get("gamma_sp_points", 24), sel)
            best = _select_config(curves, p["tol"])
            # apply the tuned config and record exact errors NOW (the model
            # is in hand), so the final assembly needs no refit.
            _apply_opm_params(m, best["gamma_sp"], gamma=best.get("gamma"))
            best = dict(best,
                        exact_n_active=int(m.n_selected_nonlinear),
                        exact_cert=float(m.certificate_),
                        exact_train=m.reconstruction_error(S_tr),
                        exact_val=m.reconstruction_error(S_va),
                        exact_test=m.reconstruction_error(S_te))
            out[skey] = best
            json.dump(out, open(OUT_JSON, "w"), indent=1)
            del m
            return None                      # one heavy stage per call
        stages[sel] = (None, None, out[skey])
    best_sel = _select_variant(stages, p["variant_tol"])
    best = stages[best_sel][2]
    for v in ("greedy", "leading"):
        out.pop(f"__stage:{key}:{v}", None)
    row = dict(
        exp=exp, deg=deg, r=p["r"], L=(None if np.isinf(L) else float(L)),
        sel=best_sel, n_active=int(best["exact_n_active"]),
        cert=float(best["exact_cert"]),
        gamma=float(best["gamma"]), gamma_sp=float(best["gamma_sp"]),
        train=float(best["exact_train"]),
        val=float(best["exact_val"]),
        test=float(best["exact_test"]))
    print(f"  [{exp}/{deg}] L={L:9.3g} sel={row['sel']:7s} "
          f"act={row['n_active']:3d} cert={row['cert']:9.3g} "
          f"train={row['train']:.3f} val={row['val']:.3f} "
          f"test={row['test']:.3f}")
    return row


def choose_L(rows, exp, deg):
    """Smallest L whose validation error is within L_TOL of the best."""
    sub = [e for e in rows if e["exp"] == exp and e["deg"] == deg]
    if not sub:
        return float("nan")
    vmin = min(e["val"] for e in sub)
    ok = [e for e in sub if e["val"] <= (1.0 + L_TOL) * vmin]
    Ls = [np.inf if e["L"] is None else e["L"] for e in ok]
    return float(min(Ls))


def plots(all_rows):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5), sharex="col")
    colors = {"p2": "#9467bd", "p23": "#d62728"}
    for j, exp in enumerate(("transport", "swe", "nse")):
        axE, axA = axes[0, j], axes[1, j]
        for deg, c in colors.items():
            sub = [e for e in all_rows
                   if e["exp"] == exp and e["deg"] == deg]
            sub.sort(key=lambda e: np.inf if e["L"] is None else e["L"])
            L = [1e7 if e["L"] is None else e["L"] for e in sub]
            axE.loglog(L, [e["test"] for e in sub], "-o", color=c,
                       label=f"test ({deg})")
            axE.loglog(L, [e["train"] for e in sub], "--s", color=c,
                       alpha=0.6, label=f"train ({deg})")
            axA.semilogx(L, [e["n_active"] for e in sub], "-o", color=c,
                         label=deg)
            Lstar = choose_L(all_rows, exp, deg)
            e_star = min(
                (e for e in sub
                 if (np.inf if e["L"] is None else e["L"]) >= Lstar),
                key=lambda e: np.inf if e["L"] is None else e["L"])
            x_star = 1e7 if e_star["L"] is None else e_star["L"]
            axE.plot(x_star, e_star["test"], "*", color=c, ms=16, mec="k",
                     mew=0.7, zorder=6)
            axA.plot(x_star, e_star["n_active"], "*", color=c, ms=16,
                     mec="k", mew=0.7, zorder=6)
        r = EXPS[exp]["r"]
        title = {"transport": "Advection-diffusion",
                 "swe": "Shallow water",
                 "nse": "Navier-Stokes (cylinder)"}[exp]
        axE.set_title(f"{title} (r={r})", fontsize=14)
        for ax in (axE, axA):
            ax.grid(True, which="both", alpha=0.3)
        axA.set_xlabel(r"Lipschitz budget $L$  ($10^7$ = no budget)",
                       fontsize=12)
        if j == 0:
            axE.set_ylabel("relative reconstruction error", fontsize=12)
            axA.set_ylabel("active nonlinear features", fontsize=12)
            axE.legend(fontsize=9)
            axA.legend(fontsize=9)
    fig.suptitle("Greedy-OPM under a prescribed Lipschitz budget: "
                 "error and feature count versus $L$ "
                 "(stars: selected per-experiment budget $L^*$)",
                 fontsize=15)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"budget_study.{ext}", dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {PLOTS/'budget_study.png'}/.pdf")


def main(phase="all"):
    PLOTS.mkdir(exist_ok=True)
    out = json.load(open(OUT_JSON)) if OUT_JSON.exists() else {}
    todo = ("transport", "swe", "nse") if phase in ("all",) else (phase,)
    t0 = time.time()
    for exp in todo:
        if exp in ("plots",):
            continue
        blocks = cfg = None
        for deg in DEGREE_SETS:
            for L in L_GRID:
                key = f"{exp}:{deg}:{L}"
                if key in out:
                    continue
                if time.time() - t0 > WALL_BUDGET:
                    print("[budget-study] wall budget hit; rerun to continue")
                    return
                if blocks is None:
                    blocks, cfg = _load(exp)
                row = None
                while row is None:
                    row = study_one(exp, deg, float(L), blocks, cfg, out, key)
                    if row is None and time.time() - t0 > WALL_BUDGET:
                        print("[budget-study] wall budget hit (staged); "
                              "rerun to continue")
                        return
                out[key] = row
                json.dump(out, open(OUT_JSON, "w"), indent=1)
    if phase in ("all", "plots"):
        all_rows = [e for k, e in out.items()
                    if not k.startswith("__stage")]
        plots(all_rows)
        for exp in ("transport", "swe", "nse"):
            for deg in DEGREE_SETS:
                print(f"  L*({exp}, {deg}) = "
                      f"{choose_L(all_rows, exp, deg):g}")


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[budget-study] elapsed {time.time()-t0:.1f}s")
