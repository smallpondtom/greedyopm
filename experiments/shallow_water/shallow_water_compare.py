"""Shallow-water 2D comparison experiment.

Generates the 2D shallow-water height field h(x, y, t) with the faithful
spectral solver (experiments/data/shallow_water.py) and compares POD,
Greedy-QM, Geelen-QM, Greedy-Fixed-OPM (p=2), Greedy-OPM (p=2), and
Greedy-OPM (p=[2,3]).  Snapshots are split TEMPORALLY into the first 50% of
timesteps (train), the next 25% (validation, near future) and the final 25%
(test, far future), so the test set is a genuine forward-in-time
extrapolation (see experiments/exp_split.py).  The validation window tunes
the Greedy-OPM hyperparameters (gamma_sp, mu); the test window is held out
for the reported errors.

The state modelled is the height field h (d = N^2); the dispersing concentric
waves give a slowly decaying Kolmogorov n-width.  The r-sweep is resumable;
rerun until DONE.  Phases: ``sweep`` | ``curves`` | ``study`` | ``active`` |
``field`` | ``all``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))             # experiments/
sys.path.insert(0, str(_HERE.parent / "data"))    # shallow_water solver
from exp_methods import run_sweep, build_model, METHOD_NAMES   # noqa: E402
from exp_plots import (error_vs_r, field_panels,               # noqa: E402
                       field_panels_horizontal, gamma_sp_study,
                       active_features_table, active_features_bar)
from exp_split import split_indices                            # noqa: E402
from shallow_water import generate_shallow_water               # noqa: E402

PLOTS = _HERE / "plots"
CKPT = PLOTS / "swe_compare_results.json"

# ---- problem + experiment configuration ------------------------------------
N = 108
# store densely: n_train must exceed the degree-<=3 feature count (<=286 at r=10)
# to keep the empirical polynomial basis well-posed (unisolvence: n >= #features).
# A broad, moderate-amplitude initial bump (ic_sharpness=1.5, bump_amp=0.25)
# excites a coherent sloshing wave field: the singular values decay slowly
# (transport-dominated n-width barrier) yet the future windows remain
# reconstructible from the training span, so the temporal train / val / test
# split is genuinely out-of-distribution without being degenerate.  Horizon
# t_end=20 lets the wavefront wrap the periodic domain ~3 times inside the
# training window.
SWE_KW = dict(N=N, dt=2.0e-3, t_end=20.0, store_every=5,
              ic_sharpness=1.5, bump_amp=0.25)
R_MAX = 10
R_FIELD = 8
RS = list(range(1, R_MAX + 1))
CFG = dict(
    s_pool=100,
    gamma_qm=1e-12,
    gamma_opm=1e-12,
    gamma_geelen=1e-8,
    geelen_rel=0.20,
    geelen_iters=400,
    tau=1e-3,
    gamma_sp_points=24,    # gamma_sp grid resolution for the validation sweep
    gamma_sp_tol=0.005,    # config band: fewest features within 0.5% of best val
    variant_tol=0.08,      # prefer leading modes unless greedy val-gain > 8%
    n_mu=14,               # mu grid resolution
    gamma_factors=(0.01, 0.1, 0.3, 1.0, 3.0, 10.0),   # ridge grid, multiples of n
    grad_dilation=0.1,     # OOD collar for the Dirichlet / certificate probes
    budget=3.0,        # per-experiment Lipschitz budget L (from budget_study)
)
TIME_BUDGET = 30.0    # per-call wall budget (sandbox 45s limit; sweep resumes)

def load_height():
    """Return height-field snapshots S_h (N*N, n_snap) and times (cached)."""
    cache = _HERE / "swe_height_cache.npz"
    if cache.exists():
        d = np.load(cache)
        return d["S_h"], {"tspan": d["tspan"]}
    S, info = generate_shallow_water(**SWE_KW)
    NN = N * N
    S_h = np.asarray(S[:NN, :])           # height field only
    np.savez(cache, S_h=S_h, tspan=np.asarray(info["tspan"]))
    return S_h, {"tspan": np.asarray(info["tspan"])}


def main(phase="all"):
    S, info = load_height()
    tspan = np.asarray(info["tspan"])
    itr, iva, ite = split_indices(S.shape[1])
    S_train, S_val, S_test = S[:, itr], S[:, iva], S[:, ite]
    print(f"[swe] N={N} d={S.shape[0]} n={S.shape[1]} "
          f"(train {S_train.shape[1]} / val {S_val.shape[1]} / "
          f"test {S_test.shape[1]})  phase={phase}")

    if phase in ("all", "sweep"):
        results, done = run_sweep(jnp.asarray(S_train), jnp.asarray(S_val),
                                  jnp.asarray(S_test), RS, CFG, str(CKPT),
                                  time_budget=TIME_BUDGET)
        if not done:
            print("[swe] sweep incomplete -- rerun to continue")
            return
        if phase == "sweep":
            print("SWEEP DONE")
            return

    import json
    results = json.load(open(CKPT))

    if phase in ("all", "curves"):
        error_vs_r(results, RS, METHOD_NAMES, str(PLOTS / "swe_error_vs_r"),
                   "Shallow water (height $h$) — reconstruction error vs reduced dimension")

    if phase in ("all", "study"):
        gamma_sp_study(results, RS, METHOD_NAMES,
                       str(PLOTS / "swe_gamma_sp_study"),
                       "Shallow water (height $h$) — gamma_sp validation study")

    if phase in ("all", "active"):
        print("\n[swe] number of active nonlinear features:")
        active_features_table(results, RS, METHOD_NAMES)
        active_features_bar(results, RS, METHOD_NAMES,
                            str(PLOTS / "swe_active_features"),
                            "Shallow water (height $h$) — active nonlinear features vs reduced dimension")

    if phase in ("all", "field"):
        ext = [-np.pi, np.pi, -np.pi, np.pi]
        # Representative late-time snapshot within each (contiguous) split:
        # take the 70% column of the window.
        disp = {"train": itr, "test": ite}
        col = {sp: int(cols[int(0.7 * len(cols))]) for sp, cols in disp.items()}
        recon = {"train": {}, "test": {}}
        rel = {"train": {}, "test": {}}
        truth = {sp: np.asarray(S[:, c]).reshape(N, N) for sp, c in col.items()}
        for nm in METHOD_NAMES:
            entry = results.get(f"{nm}@r={R_FIELD}") or {}
            model = build_model(nm, jnp.asarray(S_train), R_FIELD, CFG,
                                gamma_sp=entry.get("gamma_sp"),
                                mu=entry.get("mu"),
                                sel=entry.get("sel") or "greedy",
                                gamma=entry.get("gamma"))
            for split, c in col.items():
                truth_col = np.asarray(S[:, c])
                approx = np.asarray(model.reconstruct(jnp.asarray(S[:, c:c + 1])))[:, 0]
                recon[split][nm] = approx.reshape(N, N)
                rel[split][nm] = float(np.linalg.norm(approx - truth_col)
                                       / np.linalg.norm(truth_col))
        for split, tag in [("train", "Training"), ("test", "Test")]:
            tcol = tspan[col[split]]
            title = (f"Shallow water height $h(x,y)$ — {tag} snapshot "
                     f"(t={tcol:.2f}, r={R_FIELD})")
            # vertical (per-method) panel: narrower figure -> square cells, less
            # horizontal whitespace for the equal-aspect height field.
            field_panels(
                truth[split], recon[split], METHOD_NAMES,
                str(PLOTS / f"swe_field_{split}"), title,
                cmap="viridis", extent=ext, xlabel="$x$", ylabel="$y$",
                aspect="equal", rel_errs=rel[split],
                figsize=(6.5, 3.0 * (len(METHOD_NAMES) + 1)))
            # horizontal panel: tightly packed columns (one per method)
            field_panels_horizontal(
                truth[split], recon[split], METHOD_NAMES,
                str(PLOTS / f"swe_field_{split}_horizontal"), title,
                cmap="viridis", extent=ext, xlabel="$x$", ylabel="$y$",
                aspect="equal", rel_errs=rel[split], axes_pad=(0.05, 0.30))
    print("DONE")


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[swe] elapsed {time.time()-t0:.1f}s")
