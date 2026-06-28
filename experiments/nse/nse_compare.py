"""Transient Navier-Stokes (Karman vortex, Re=30) comparison experiment.

Runs POD, Greedy-QM, Geelen-QM, Greedy-Fixed-OPM (p=2), Greedy-OPM (p=2), and
Greedy-OPM (p=[2,3]) on the compressed velocity snapshots (even = train, odd =
test).  Reports the full-state relative reconstruction error (the K-POD floor is
folded back in) and the sensor-output relative error, and plots the sensor
outputs y(t) = C x for each method against the ground truth.

Phases (each < ~35s for the sandbox limit; sweep is resumable):
    sweep | curves | timeseries | all
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
sys.path.insert(0, str(_HERE.parent))                              # experiments/
DATA_DIR = _HERE / "data"
from exp_methods import build_model, METHOD_NAMES, METHOD_STYLES   # noqa: E402
from exp_plots import (error_vs_r,                                 # noqa: E402
                       active_features_table, active_features_bar)

DATA = DATA_DIR / "nse_compressed.npz"
PLOTS = _HERE / "plots"
CKPT = PLOTS / "nse_compare_results.json"

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
    tau=1e-3
)
TIME_BUDGET = 300


def load():
    d = np.load(DATA)
    return {k: d[k] for k in d.files}


def _metrics(model, A, Y, floor, nrm, C_red, y_ref):
    A = jnp.asarray(A)
    Ah = np.asarray(model.reconstruct(A))
    state_abs = float(np.linalg.norm(np.asarray(A) - Ah))
    state_rel = float(np.sqrt(state_abs**2 + floor**2) / nrm)  # full-state error
    y = y_ref[:, None] + C_red @ Ah
    out_rel = float(np.linalg.norm(Y - y) / np.linalg.norm(Y))
    return state_rel, out_rel


def _n_active(model):
    for a in ("n_selected_nonlinear", "n_active"):
        if hasattr(model, a):
            return int(getattr(model, a))
    return -1


def run_sweep(D, budget=33.0):
    res = json.load(open(CKPT)) if CKPT.exists() else {}
    A_tr, A_te = D["A_train"], D["A_test"]
    Y_tr, Y_te = D["Y_train"], D["Y_test"]
    C_red, y_ref = D["C_red"], D["y_ref"]
    ftr, fte = float(D["floor_train"]), float(D["floor_test"])
    ntr, nte = float(D["nrm_train"]), float(D["nrm_test"])
    t0 = time.time(); did = False
    for r in RS:
        for nm in METHOD_NAMES:
            key = f"{nm}@r={r}"
            if key in res:
                continue
            if did and time.time() - t0 > budget:
                _save(res); print("[budget] pausing; rerun to continue"); return res, False
            tic = time.time()
            model = build_model(nm, jnp.asarray(A_tr), r, CFG)
            fit_secs = round(time.time() - tic, 3)     # fit-only wall time
            s_tr, o_tr = _metrics(model, A_tr, Y_tr, ftr, ntr, C_red, y_ref)
            s_te, o_te = _metrics(model, A_te, Y_te, fte, nte, C_red, y_ref)
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
    A_tr, A_te = jnp.asarray(D["A_train"]), jnp.asarray(D["A_test"])
    Y_tr, Y_te = D["Y_train"], D["Y_test"]
    C_red, y_ref = D["C_red"], D["y_ref"]
    t_tr, t_te = D["trange_train"], D["trange_test"]
    n_out = Y_tr.shape[0]
    ntot = Y_tr.shape[1] + Y_te.shape[1]

    tfull = np.empty(ntot); tfull[::2] = t_tr; tfull[1::2] = t_te
    Ytrue = np.empty((n_out, ntot)); Ytrue[:, ::2] = Y_tr; Ytrue[:, 1::2] = Y_te

    ymethods = {}
    for nm in METHOD_NAMES:
        model = build_model(nm, A_tr, R_OUT, CFG)
        ye = y_ref[:, None] + C_red @ np.asarray(model.reconstruct(A_tr))
        yo = y_ref[:, None] + C_red @ np.asarray(model.reconstruct(A_te))
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
    fig.suptitle(f"Karman vortex — sensor outputs $y(t)=Cx$  (r={R_OUT}, "
                 f"reconstruction of full trajectory)", fontsize=18, y=1.0)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(PLOTS / f"nse_output_timeseries.{ext}"), dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print("    saved nse_output_timeseries.png/.pdf")


def main(phase="all"):
    D = load()
    print(f"[nse] K={int(D['K'])} train {D['A_train'].shape} test {D['A_test'].shape}"
          f"  n_outputs={D['Y_train'].shape[0]}  phase={phase}")
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
        error_vs_r(res_state, RS, METHOD_NAMES, str(PLOTS / "nse_state_error_vs_r"),
                   "Karman vortex — full-state reconstruction error vs reduced dimension")
        error_vs_r(res_out, RS, METHOD_NAMES, str(PLOTS / "nse_output_error_vs_r"),
                   "Karman vortex — sensor-output error $\\|y-\\hat y\\|/\\|y\\|$ vs reduced dimension")
    if phase in ("all", "active"):
        print("\n[nse] number of active nonlinear features:")
        active_features_table(res, RS, METHOD_NAMES)
        active_features_bar(res, RS, METHOD_NAMES,
                            str(PLOTS / "nse_active_features"),
                            "Karman vortex — active nonlinear features vs reduced dimension")
    if phase in ("all", "timeseries"):
        timeseries(D)
    print("DONE")


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[nse] elapsed {time.time()-t0:.1f}s")
