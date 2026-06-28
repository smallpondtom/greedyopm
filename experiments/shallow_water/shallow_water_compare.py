"""Shallow-water 2D comparison experiment.

Generates the 2D shallow-water height field h(x, y, t) with the faithful
spectral solver (experiments/data/shallow_water.py), uses even snapshots for
training and odd for testing, and compares POD, Greedy-QM, Geelen-QM,
Greedy-Fixed-OPM (p=2), Greedy-OPM (p=2), and Greedy-OPM (p=[2,3]).

The state modelled is the height field h (d = N^2); the dispersing concentric
waves give a slowly decaying Kolmogorov n-width.  The r-sweep is resumable;
rerun until DONE.  Phases: ``sweep`` | ``curves`` | ``field`` | ``all``.
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
                       field_panels_horizontal,
                       active_features_table, active_features_bar)
from shallow_water import generate_shallow_water               # noqa: E402

PLOTS = _HERE / "plots"
CKPT = PLOTS / "swe_compare_results.json"

# ---- problem + experiment configuration ------------------------------------
N = 64
# store densely: n_train must exceed the degree-<=3 feature count (<=286 at r=10)
# to keep the empirical polynomial basis well-posed (unisolvence: n >= #features).
SWE_KW = dict(N=N, dt=2.0e-3, t_end=8.0, store_every=2, ic_sharpness=4.0)
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
    tau=1e-3
)
TIME_BUDGET = 300.0

def load_height():
    """Return height-field snapshots S_h (N*N, n_snap) and grids."""
    S, info = generate_shallow_water(**SWE_KW)
    NN = N * N
    S_h = S[:NN, :]                       # height field only
    return S_h, info


def main(phase="all"):
    S, info = load_height()
    S_train, S_test = S[:, ::2], S[:, 1::2]
    tspan = info["tspan"]
    print(f"[swe] N={N} d={S.shape[0]} n={S.shape[1]} "
          f"(train {S_train.shape[1]} / test {S_test.shape[1]})  phase={phase}")

    if phase in ("all", "sweep"):
        results, done = run_sweep(jnp.asarray(S_train), jnp.asarray(S_test),
                                  RS, CFG, str(CKPT), time_budget=TIME_BUDGET)
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

    if phase in ("all", "active"):
        print("\n[swe] number of active nonlinear features:")
        active_features_table(results, RS, METHOD_NAMES)
        active_features_bar(results, RS, METHOD_NAMES,
                            str(PLOTS / "swe_active_features"),
                            "Shallow water (height $h$) — active nonlinear features vs reduced dimension")

    if phase in ("all", "field"):
        ext = [-np.pi, np.pi, -np.pi, np.pi]
        # representative late-time snapshots (waves dispersed)
        col = {"train": int(0.7 * S_train.shape[1]),
               "test": int(0.7 * S_test.shape[1])}
        recon = {"train": {}, "test": {}}
        rel = {"train": {}, "test": {}}
        truth = {}
        for split, Smat in [("train", S_train), ("test", S_test)]:
            truth[split] = np.asarray(Smat[:, col[split]]).reshape(N, N)
        for nm in METHOD_NAMES:
            model = build_model(nm, jnp.asarray(S_train), R_FIELD, CFG)
            for split, Smat in [("train", S_train), ("test", S_test)]:
                approx = np.asarray(model.reconstruct(jnp.asarray(Smat)))
                recon[split][nm] = approx[:, col[split]].reshape(N, N)
                rel[split][nm] = float(
                    np.linalg.norm(approx[:, col[split]] - np.asarray(Smat[:, col[split]]))
                    / np.linalg.norm(np.asarray(Smat[:, col[split]])))
        for split, tag in [("train", "Training"), ("test", "Test")]:
            tcol = tspan[::2][col["train"]] if split == "train" else tspan[1::2][col["test"]]
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
