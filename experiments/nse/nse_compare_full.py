"""Transient Navier-Stokes (Karman vortex, Re=30) comparison -- FULL data.

Identical experiment to ``nse_compare.py`` but every method is fit directly on
the full ``d = 41682``-dimensional velocity snapshots, with NO method-of-snapshots
POD compression (see ``prep_nse.py`` / ``DATA_PREPARATION.md`` for the compressed
pipeline).  This is the expensive baseline: because the compared methods are
isometry-invariant, the reported errors match the compressed run to the POD
truncation tolerance -- only the *runtime* differs.

Even columns = train, odd = test (same split as the compressed run, using the
*training* mean for centering).  The reported errors are

    full-state :  ||V - V_hat||_F / ||V - s_ref||_F      (no POD floor -- nothing
                  is discarded, so there is no out-of-span residual to fold in)
    output     :  ||Y - C V_hat||_F / ||Y||_F ,  Y = C V .

Outputs are written with a ``_full`` suffix so they sit next to the compressed
run without clobbering it:

    plots/nse_compare_results_full.json          (resumable checkpoint)
    plots/nse_state_error_vs_r_full.{png,pdf}
    plots/nse_output_error_vs_r_full.{png,pdf}
    plots/nse_active_features_full.{png,pdf}
    plots/nse_output_timeseries_full.{png,pdf}

Phases:  sweep | curves | active | timeseries | all   (sweep is resumable; set
NSE_FULL_BUDGET to a wall-clock budget in seconds to pause/resume across runs).
"""

from __future__ import annotations

import json
import os
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
sys.path.insert(0, str(_HERE.parent))                              # experiments/
DATA_DIR = _HERE / "data"
from exp_methods import build_model, METHOD_NAMES, METHOD_STYLES   # noqa: E402
from exp_plots import (error_vs_r,                                 # noqa: E402
                       active_features_table, active_features_bar)

RAW = DATA_DIR / "nse_snapshots.npz"
PLOTS = _HERE / "plots"
CKPT = PLOTS / "nse_compare_results_full.json"

R_MAX = 10
R_OUT = 10               # r used for the y(t) timeseries (matches runme rv=10)
RS = list(range(1, R_MAX + 1))
CFG = dict(
    s_pool=100,
    gamma_qm=1e-12,
    gamma_opm=1e-12,
    gamma_geelen=1e-8,
    geelen_rel=0.10,
    geelen_iters=400,
    tau=1e-3,
)
# Full-data fits are expensive; default to "no budget" (run to completion in one
# call), but allow a finite wall-clock budget for resumable runs.
TIME_BUDGET = float(os.environ.get("NSE_FULL_BUDGET", "1e9"))


def load():
    """Load the raw snapshots and derive the train/test split + output matrix."""
    d = np.load(RAW)
    V = d["V"].astype(np.float64)            # (d, 2000)
    trange = d["trange"]
    C = d["C"].astype(np.float64)            # (6, d)
    V_tr, V_te = V[:, ::2], V[:, 1::2]
    t_tr, t_te = trange[::2], trange[1::2]
    s_ref = V_tr.mean(axis=1, keepdims=True)             # training mean
    nrm_tr = float(np.linalg.norm(V_tr - s_ref))         # centered training norm
    nrm_te = float(np.linalg.norm(V_te - s_ref))         # centered test norm
    Y_tr = C @ V_tr                                      # (6, ntr)
    Y_te = C @ V_te                                      # (6, nte)
    return dict(V_tr=V_tr, V_te=V_te, Y_tr=Y_tr, Y_te=Y_te,
                t_tr=t_tr, t_te=t_te, nrm_tr=nrm_tr, nrm_te=nrm_te,
                C=C, d=int(V.shape[0]))


def _metrics(model, S, Y, nrm, C):
    """Full-state + sensor-output relative errors on the raw d-dim snapshots."""
    Sh = np.asarray(model.reconstruct(jnp.asarray(S)))   # (d, n)
    state_rel = float(np.linalg.norm(np.asarray(S) - Sh) / nrm)   # no POD floor
    y = C @ Sh                                           # (6, n)
    out_rel = float(np.linalg.norm(Y - y) / np.linalg.norm(Y))
    return state_rel, out_rel


def _n_active(model):
    for a in ("n_selected_nonlinear", "n_active"):
        if hasattr(model, a):
            return int(getattr(model, a))
    return -1


def _block(model):
    """Force JAX to finish the fit before stopping the timer.

    JAX dispatches asynchronously, so a purely lazy fit (notably POD: mean ->
    SVD -> slice, with no host-side data dependency) returns before the SVD has
    actually run, and its cost would otherwise be mis-attributed to the later
    metrics step.  Blocking on the fitted arrays makes ``fit_secs`` honest for
    every method.
    """
    arrs = [v for v in vars(model).values() if isinstance(v, jnp.ndarray)]
    if arrs:
        jax.block_until_ready(arrs)
    return model


def run_sweep(D, budget=TIME_BUDGET):
    res = json.load(open(CKPT)) if CKPT.exists() else {}
    V_tr, V_te = D["V_tr"], D["V_te"]
    Y_tr, Y_te = D["Y_tr"], D["Y_te"]
    C = D["C"]
    ntr, nte = D["nrm_tr"], D["nrm_te"]
    V_tr_j = jnp.asarray(V_tr)                  # device once (not per-fit)
    t0 = time.time(); did = False
    for r in RS:
        for nm in METHOD_NAMES:
            key = f"{nm}@r={r}"
            if key in res:
                continue
            if did and time.time() - t0 > budget:
                _save(res); print("[budget] pausing; rerun to continue"); return res, False
            tic = time.time()
            model = _block(build_model(nm, V_tr_j, r, CFG))
            fit_secs = round(time.time() - tic, 3)     # fit-only wall time (blocked)
            s_tr, o_tr = _metrics(model, V_tr, Y_tr, ntr, C)
            s_te, o_te = _metrics(model, V_te, Y_te, nte, C)
            res[key] = dict(method=nm, r=int(r), state_train=s_tr, state_test=s_te,
                            out_train=o_tr, out_test=o_te, n_active=_n_active(model),
                            fit_secs=fit_secs,
                            secs=round(time.time() - tic, 2))
            did = True
            print(f"  r={r:2d} {nm:24s} state(te)={s_te:.3e} out(te)={o_te:.3e} "
                  f"act={res[key]['n_active']:>3d} ({res[key]['secs']}s)")
            _save(res)
    _save(res); return res, True


def _save(res):
    PLOTS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(CKPT, "w"), indent=0)


def timeseries(D):
    """y(t) = C x for each method vs ground truth, full interleaved trajectory."""
    V_tr, V_te = jnp.asarray(D["V_tr"]), jnp.asarray(D["V_te"])
    Y_tr, Y_te = D["Y_tr"], D["Y_te"]
    C = D["C"]
    t_tr, t_te = D["t_tr"], D["t_te"]
    n_out = Y_tr.shape[0]
    ntot = Y_tr.shape[1] + Y_te.shape[1]

    tfull = np.empty(ntot); tfull[::2] = t_tr; tfull[1::2] = t_te
    Ytrue = np.empty((n_out, ntot)); Ytrue[:, ::2] = Y_tr; Ytrue[:, 1::2] = Y_te

    ymethods = {}
    for nm in METHOD_NAMES:
        model = build_model(nm, V_tr, R_OUT, CFG)
        ye = C @ np.asarray(model.reconstruct(V_tr))
        yo = C @ np.asarray(model.reconstruct(V_te))
        yf = np.empty((n_out, ntot)); yf[:, ::2] = ye; yf[:, 1::2] = yo
        ymethods[nm] = yf

    fig, axes = plt.subplots(3, 2, figsize=(18, 11), sharex=True)
    axes = axes.ravel()
    for i in range(n_out):
        ax = axes[i]
        ax.plot(tfull, Ytrue[i], color="k", lw=2.4, label="Ground truth", zorder=10)
        for nm in METHOD_NAMES:
            st = METHOD_STYLES[nm]
            ls = "--" if "--" in st["fmt"] else "-"
            ax.plot(tfull, ymethods[nm][i], ls, color=st["color"],
                    lw=1.4, alpha=0.9, label=nm)
        ax.set_title(f"sensor output $y_{i+1}(t)$", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        if i >= 4:
            ax.set_xlabel("time $t$", fontsize=13)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=13,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle(f"Karman vortex (full data) — sensor outputs $y(t)=Cx$  (r={R_OUT}, "
                 f"reconstruction of full trajectory)", fontsize=18, y=1.0)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(PLOTS / f"nse_output_timeseries_full.{ext}"), dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print("    saved nse_output_timeseries_full.png/.pdf")


def main(phase="all"):
    D = load()
    print(f"[nse-full] d={D['d']} train {D['V_tr'].shape} test {D['V_te'].shape}"
          f"  n_outputs={D['Y_tr'].shape[0]}  phase={phase}")
    if phase in ("all", "sweep"):
        res, done = run_sweep(D, budget=TIME_BUDGET)
        if not done:
            return
        if phase == "sweep":
            print("SWEEP DONE"); return
    res = json.load(open(CKPT))

    if phase in ("all", "curves"):
        res_state = {k: {"train": v["state_train"], "test": v["state_test"]}
                     for k, v in res.items()}
        res_out = {k: {"train": v["out_train"], "test": v["out_test"]}
                   for k, v in res.items()}
        error_vs_r(res_state, RS, METHOD_NAMES, str(PLOTS / "nse_state_error_vs_r_full"),
                   "Karman vortex (full data) — full-state reconstruction error vs reduced dimension")
        error_vs_r(res_out, RS, METHOD_NAMES, str(PLOTS / "nse_output_error_vs_r_full"),
                   "Karman vortex (full data) — sensor-output error $\\|y-\\hat y\\|/\\|y\\|$ vs reduced dimension")
    if phase in ("all", "active"):
        print("\n[nse-full] number of active nonlinear features:")
        active_features_table(res, RS, METHOD_NAMES)
        active_features_bar(res, RS, METHOD_NAMES,
                            str(PLOTS / "nse_active_features_full"),
                            "Karman vortex (full data) — active nonlinear features vs reduced dimension")
    if phase in ("all", "timeseries"):
        timeseries(D)
    print("DONE")


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[nse-full] elapsed {time.time()-t0:.1f}s")
