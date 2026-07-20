"""Transient Navier-Stokes (Karman vortex, Re=30) comparison experiment.

Runs POD, Greedy-QM, Geelen-QM, Greedy-Fixed-OPM (p=2), Greedy-OPM (p=2), and
Greedy-OPM (p=[2,3]) on the compressed velocity snapshots.  The snapshots are
split TEMPORALLY (first 50% train / next 25% validation / final 25% test, see
experiments/exp_split.py and nse/prep_nse.py), so the test window is a
genuine forward-in-time extrapolation.  The validation window tunes the
Greedy-OPM hyperparameters (gamma, gamma_sp, mode-selection variant); the test
window is held out for the reported errors.  Reports the full-state relative
reconstruction error (the K-POD floor is folded back in) and the
sensor-output relative error, and plots the sensor outputs y(t) = C x for
each method against the ground truth.

Phases (each < ~35s for the sandbox limit; sweep is resumable):
    sweep | curves | active | timeseries | all
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
from exp_methods import (build_model, fit_opm_tuned,               # noqa: E402
                         METHOD_NAMES, METHOD_STYLES,
                         _OPM_METHODS, _OPM_DEGREES)
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
    tau=1e-3,
    gamma_sp_points=24,   # gamma_sp grid resolution for the validation sweep
    gamma_sp_tol=0.02,    # within-tol selection (best train, then fewest feats)
    grad_dilation=0.1,    # OOD collar for the Dirichlet / certificate probes
    n_grad_samples=150,   # in-loop gradient-probe subsample (final pass = full cloud)
    budget=316.0,        # per-experiment Lipschitz budget L (from budget_study)
    gamma_factors=(0.01, 0.1, 0.3, 1.0, 3.0, 10.0),  # ridge grid, multiples of n
)
TIME_BUDGET = 30.0


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
    A_tr, A_va, A_te = D["A_train"], D["A_val"], D["A_test"]
    C_red, y_ref = D["C_red"], D["y_ref"]
    splits = {
        "train": (A_tr, D["Y_train"], float(D["floor_train"]),
                  float(D["nrm_train"])),
        "val": (A_va, D["Y_val"], float(D["floor_val"]), float(D["nrm_val"])),
        "test": (A_te, D["Y_test"], float(D["floor_test"]),
                 float(D["nrm_test"])),
    }
    t0 = time.time(); did = False
    for r in RS:
        for nm in METHOD_NAMES:
            key = f"{nm}@r={r}"
            if key in res:
                continue
            if did and time.time() - t0 > budget:
                _save(res); print("[budget] pausing; rerun to continue"); return res, False
            tic = time.time()
            if nm in _OPM_METHODS:
                model, gsp, _, sel, gam = fit_opm_tuned(
                    jnp.asarray(A_tr), jnp.asarray(A_va), r,
                    _OPM_DEGREES[nm], CFG)
            else:
                model, gsp, sel, gam = (
                    build_model(nm, jnp.asarray(A_tr), r, CFG),
                    None, None, None)
            fit_secs = round(time.time() - tic, 3)     # fit + tuning wall time
            entry = dict(method=nm, r=int(r), n_active=_n_active(model),
                         gamma_sp=(None if gsp is None else float(gsp)),
                         gamma=(None if gam is None else float(gam)),
                         cert=(None if getattr(model, "certificate_", None)
                               is None else float(model.certificate_)), sel=sel,
                         fit_secs=fit_secs)
            for tag, (A, Y, floor, nrm) in splits.items():
                s_rel, o_rel = _metrics(model, A, Y, floor, nrm, C_red, y_ref)
                entry[f"state_{tag}"] = s_rel
                entry[f"out_{tag}"] = o_rel
            entry["secs"] = round(time.time() - tic, 2)
            res[key] = entry
            did = True
            ce = entry.get("cert")
            stxt = (f" gsp={gsp:.2e} cert={ce:.3g} sel={sel}"
                    if gsp is not None else "")
            print(f"  r={r:2d} {nm:24s} state(tr)={entry['state_train']:.3e} "
                  f"state(te)={entry['state_test']:.3e} "
                  f"act={entry['n_active']:>3d}{stxt} ({entry['secs']}s)")
            _save(res)
    _save(res); return res, True


def _save(res):
    PLOTS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(CKPT, "w"), indent=0)


def timeseries(D, res):
    """y(t) = C x for each method vs ground truth over the full horizon."""
    C_red, y_ref = D["C_red"], D["y_ref"]
    A = {t: jnp.asarray(D[f"A_{t}"]) for t in ("train", "val", "test")}
    t = {t_: D[f"trange_{t_}"] for t_ in ("train", "val", "test")}
    order = ("train", "val", "test")
    tfull = np.concatenate([t[k] for k in order])
    Ytrue = np.concatenate([D[f"Y_{k}"] for k in order], axis=1)
    n_out = Ytrue.shape[0]

    # Per-method y(t) cache so the phase is resumable under the 45s limit.
    ycache = PLOTS / "nse_timeseries_cache.npz"
    ymethods = dict(np.load(ycache)) if ycache.exists() else {}
    ts_methods = [m for m in METHOD_NAMES
                  if m != "Greedy-OPM (p=[2,3,4])"]
    for nm in ts_methods:
        if nm in ymethods:
            continue
        entry = res.get(f"{nm}@r={R_OUT}") or {}
        model = build_model(nm, A["train"], R_OUT, CFG,
                            gamma_sp=entry.get("gamma_sp"),
                            sel=entry.get("sel") or "greedy",
                            gamma=entry.get("gamma"))
        yy = [y_ref[:, None] + C_red @ np.asarray(model.reconstruct(A[k]))
              for k in order]
        ymethods[nm] = np.concatenate(yy, axis=1)
        np.savez(ycache, **ymethods)
        print(f"    cached y(t) for {nm}")

    fig, axes = plt.subplots(3, 2, figsize=(18, 11), sharex=True)
    axes = axes.ravel()
    t_train_end = t["train"][-1]
    t_val_end = t["val"][-1]
    for i in range(n_out):
        ax = axes[i]
        ax.plot(tfull, Ytrue[i], color="k", lw=2.4, label="Ground truth", zorder=10)
        for nm in ts_methods:
            st = METHOD_STYLES[nm]
            ls = "--" if "--" in st["fmt"] else "-"
            ax.plot(tfull, ymethods[nm][i], ls, color=st["color"],
                    lw=1.4, alpha=0.9, label=nm)
        for tv, lab in ((t_train_end, "train|val"), (t_val_end, "val|test")):
            ax.axvline(tv, color="gray", lw=1.0, ls=":", zorder=5)
        ax.set_title(f"sensor output $y_{i+1}(t)$", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        if i >= 4:
            ax.set_xlabel("time $t$", fontsize=13)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=13,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("Karman vortex — sensor outputs $y(t)=Cx$  "
                 f"(r={R_OUT}; dotted lines mark the train/val/test windows)",
                 fontsize=18, y=1.0)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(PLOTS / f"nse_output_timeseries.{ext}"), dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print("    saved nse_output_timeseries.png/.pdf")


def main(phase="all"):
    D = load()
    print(f"[nse] K={int(D['K'])} train {D['A_train'].shape} "
          f"val {D['A_val'].shape} test {D['A_test'].shape}"
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
        timeseries(D, res)
    print("DONE")


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[nse] elapsed {time.time()-t0:.1f}s")
