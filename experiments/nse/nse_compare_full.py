"""Transient Navier-Stokes (Karman vortex, Re=30) comparison -- FULL data.

Identical experiment to ``nse_compare.py`` (same temporal 70/15/15 split,
same method settings and validation tuning), but every method is fit
directly on the full ``d = 41682``-dimensional velocity snapshots, with NO
method-of-snapshots POD compression.  Because the compared methods are
isometry-invariant, the reported errors match the compressed run up to the
POD truncation tolerance -- but the runtimes are the honest full-state
costs, and the reported errors carry no truncation floor.

The reported errors are

    full-state :  ||V - V_hat||_F / ||V - s_ref||_F
    output     :  ||Y - C V_hat||_F / ||Y||_F ,  Y = C V .

Outputs are written with a ``_full`` suffix so they sit next to the
compressed run without clobbering it:

    plots/nse_compare_results_full.json          (resumable checkpoint)
    plots/nse_state_error_vs_r_full.{png,pdf}
    plots/nse_output_error_vs_r_full.{png,pdf}
    plots/nse_active_features_full.{png,pdf}
    plots/nse_output_timeseries_full.{png,pdf}

Phases:  sweep | curves | active | timeseries | all   (sweep is resumable;
set NSE_FULL_BUDGET to a wall-clock budget in seconds to pause/resume).
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
from exp_methods import (build_model, METHOD_NAMES, METHOD_STYLES,  # noqa: E402
                         fit_opm_tuned, _OPM_METHODS, _OPM_DEGREES,
                         _active_by_degree, _fit_opm_base, _grid_curves,
                         _grid_curves_budget, _apply_budget_config,
                         _select_config, _select_variant, _apply_opm_params)
from exp_plots import (error_vs_r,                                 # noqa: E402
                       active_features_table, active_features_bar)
from exp_split import split_indices                                # noqa: E402

RAW = DATA_DIR / "nse_snapshots.npz"
PLOTS = _HERE / "plots"
CKPT = PLOTS / "nse_compare_results_full.json"

R_MAX = 10
R_OUT = 10               # r used for the y(t) timeseries (matches runme rv=10)
RS = list(range(1, R_MAX + 1))
# Same settings as the data-efficient (compressed) script nse_compare.py.
CFG = dict(
    s_pool=100,
    gamma_qm=1e-12,
    gamma_opm=1e-12,
    gamma_geelen=1e-8,
    geelen_rel=0.10,
    geelen_iters=400,
    tau=1e-3,
    gamma_sp_points=24,   # gamma_sp grid resolution for the validation sweep
    gamma_sp_tol=0.02,    # config band: fewest features within 2% of best val
    n_mu=14,              # mu grid resolution
    grad_dilation=0.1,    # OOD collar for the Dirichlet / certificate probes
    budget=1e3,        # per-experiment Lipschitz budget L (from budget_study)
    gamma_factors=(0.01, 0.1, 0.3, 1.0, 3.0, 10.0),  # ridge grid, multiples of n
)
# Full-data fits are expensive; allow a finite wall-clock budget so the sweep
# can pause/resume across sandbox calls.
TIME_BUDGET = float(os.environ.get("NSE_FULL_BUDGET", "30"))


def load():
    """Raw snapshots + temporal 70/15/15 split (same as nse/prep_nse.py).

    The 70% training fraction reaches the saturated vortex-shedding limit
    cycle; validation (near future) tunes the Greedy-OPM hyperparameters and
    the final 15% (far future) is the held-out test window.
    """
    d = np.load(RAW)
    V = d["V"]                               # (d, 2000), float64 on disk
    if V.dtype != np.float64:
        V = V.astype(np.float64)
    trange = np.asarray(d["trange"])
    C = d["C"].astype(np.float64)            # (6, d)
    itr, iva, ite = split_indices(V.shape[1], train_frac=0.7, val_frac=0.15)
    t_tr, t_va, t_te = trange[itr], trange[iva], trange[ite]
    Y = C @ V                                # (6, 2000), cheap
    # The temporal split blocks are contiguous: slice by VIEW and copy each
    # block straight to the device -- exactly ONE copy of each split lives in
    # memory (the sandbox has ~3 GB of RAM; the raw matrix is 0.67 GB).
    sl = {k: slice(int(idx[0]), int(idx[-1]) + 1)
          for k, idx in (("tr", itr), ("va", iva), ("te", ite))}
    out = dict(t_tr=t_tr, t_va=t_va, t_te=t_te, C=C, d=int(V.shape[0]),
               Y_tr=Y[:, itr], Y_va=Y[:, iva], Y_te=Y[:, ite])
    s_ref = None
    nrm = {}
    for key in ("tr", "va", "te"):
        blk = jnp.asarray(V[:, sl[key]])
        if s_ref is None:                    # training mean, from the block
            s_ref = jnp.mean(blk, axis=1, keepdims=True)
        nrm[key] = float(jnp.linalg.norm(blk - s_ref))
        out[f"V_{key}"] = blk
    del V, d
    out["nrm"] = dict(train=nrm["tr"], val=nrm["va"], test=nrm["te"])
    return out


def _metrics(model, S, Y, nrm, C):
    """Full-state + sensor-output relative errors on the raw d-dim snapshots.

    ``S`` is a device array; the reconstruction is consumed on device to
    avoid a second host copy (memory-constrained sandbox).
    """
    Sh = model.reconstruct(S)                            # (d, n), device
    state_rel = float(jnp.linalg.norm(S - Sh) / nrm)     # no POD floor
    y = jnp.asarray(C) @ Sh                              # (6, n)
    out_rel = float(jnp.linalg.norm(jnp.asarray(Y) - y)
                    / np.linalg.norm(Y))
    return state_rel, out_rel


def _n_active(model):
    for a in ("n_selected_nonlinear", "n_active"):
        if hasattr(model, a):
            return int(getattr(model, a))
    return -1


def _block(model):
    """Force JAX to finish the fit before stopping the timer.

    JAX dispatches asynchronously, so a purely lazy fit (notably POD) would
    otherwise return before the SVD has actually run and its cost would be
    mis-attributed to the later metrics step.
    """
    arrs = [v for v in vars(model).values() if isinstance(v, jnp.ndarray)]
    if arrs:
        jax.block_until_ready(arrs)
    return model


SVD_CACHE = DATA_DIR / "nse_full_svd_cache.npz"
SVD_CACHE_FIT = DATA_DIR / "nse_full_svd_trainval_cache.npz"


def _shared_svd(V_blk, cache):
    """Memory-lean rank-s_pool SVD (method of snapshots), cached to disk.

    Never forms a d x n factor: eigendecompose the (n x n) Gram matrix and
    lift only the leading s_pool modes, ``U_s = X W_s / sigma_s``.  Cached so
    resumed sweep calls skip the recomputation.
    """
    if cache.exists():
        z = np.load(cache)
        return z["U_s"], z["sigma"], z["W_s"]
    X = V_blk - jnp.mean(V_blk, axis=1, keepdims=True)
    G = np.asarray(X.T @ X)                      # (n, n)
    w, W = np.linalg.eigh(G)
    w = np.clip(w[::-1], 0.0, None); W = np.ascontiguousarray(W[:, ::-1])
    k = int(CFG["s_pool"])
    sigma = np.sqrt(w[:k]); W_s = W[:, :k]
    U_s = np.asarray(X @ jnp.asarray(W_s / sigma[None, :]))   # (d, k)
    np.savez(cache, U_s=U_s, sigma=sigma, W_s=W_s)
    return U_s, sigma, W_s


def _staged_opm(key, res, V_tr, V_va, r, degrees):
    """Validation-tuned OPM fit, STAGED across sandbox calls.

    The two mode-selection variants are tuned in separate stages whose
    (JSON-serializable) results are checkpointed under ``__stage:`` keys, so
    each call needs to fit at most one greedy variant; the final call refits
    the selected variant and returns the model.  Mirrors
    ``exp_methods.fit_opm_tuned`` exactly.
    """
    exact = CFG.get("exact_polish", False)
    n_points = CFG.get("gamma_sp_points", 24)
    tol = CFG.get("gamma_sp_tol", 0.02)
    stages = {}
    stage_secs = 0.0
    for sel, s_pool in (("greedy", None), ("leading", r)):
        skey = f"__stage:{key}:{sel}"
        if skey not in res:
            tic = time.time()
            m = _fit_opm_base(V_tr, r, degrees, CFG, s_pool=s_pool)
            L = CFG.get("budget")
            if L is not None:
                curves = _grid_curves_budget(m, V_tr, V_va, CFG, n_points,
                                             sel, float(L))
            else:
                curves = _grid_curves(m, V_tr, V_va, CFG, n_points, exact,
                                      sel)
            best = _select_config(curves, tol)
            if L is not None:                 # budget mode: mu varies per point
                curve = next(c for c in curves
                             if c and c[0].get("gamma") == best.get("gamma"))
            else:
                curve = next(c for c in curves
                             if c and c[0]["mu"] == best["mu"]
                             and c[0].get("gamma") == best.get("gamma"))
            res[skey] = dict(best=best, curve=curve,
                             secs=round(time.time() - tic, 3))
            _save(res)
            del m
            return None                      # one heavy stage per call
        stage_secs += float(res[skey].get("secs", 0.0))
        stages[sel] = (None, res[skey]["curve"], res[skey]["best"])

    best_sel = _select_variant(stages, CFG.get("variant_tol", tol))
    _, curve, best = stages[best_sel]
    m = _fit_opm_base(V_tr, r, degrees, CFG,
                      s_pool=(r if best_sel == "leading" else None))
    L = CFG.get("budget")
    if L is not None:
        _apply_budget_config(m, best, float(L))
    else:
        _apply_opm_params(m, best["gamma_sp"], best["mu"], exact,
                          gamma=best.get("gamma"))
    for sel in ("greedy", "leading"):
        res.pop(f"__stage:{key}:{sel}", None)
    return (m, best["gamma_sp"], best["mu"], curve, best_sel,
            best.get("gamma"), stage_secs)


def run_sweep(D, budget=TIME_BUDGET):
    res = json.load(open(CKPT)) if CKPT.exists() else {}
    C = D["C"]
    V_tr_j, V_va_j = D["V_tr"], D["V_va"]       # already device arrays
    CFG["_svd"] = _shared_svd(V_tr_j, SVD_CACHE)             # all OPM fits
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
                out = _staged_opm(key, res, V_tr_j, V_va_j, r,
                                  _OPM_DEGREES[nm])
                if out is None:               # stage checkpointed; next call
                    print(f"  r={r:2d} {nm:24s} [staged; rerun to continue]")
                    did = True
                    continue
                model, gsp, mu, curve, sel, gam, stage_secs = out
                _block(model)
                tic -= stage_secs             # include the staged tuning time
            else:
                model = _block(build_model(nm, V_tr_j, r, CFG))
                gsp, mu, curve, sel, gam = None, None, None, None, None
            fit_secs = round(time.time() - tic, 3)     # fit + tuning wall time
            s_tr, o_tr = _metrics(model, D["V_tr"], D["Y_tr"], D["nrm"]["train"], C)
            s_va, o_va = _metrics(model, D["V_va"], D["Y_va"], D["nrm"]["val"], C)
            s_te, o_te = _metrics(model, D["V_te"], D["Y_te"], D["nrm"]["test"], C)
            n_act = _n_active(model)
            act_by_deg = _active_by_degree(model)
            cert_val = getattr(model, "certificate_", None)
            del model                                   # free d-dim arrays
            res[key] = dict(method=nm, r=int(r),
                            state_train=s_tr, state_val=s_va, state_test=s_te,
                            out_train=o_tr, out_val=o_va, out_test=o_te,
                            n_active=n_act,
                            n_active_by_degree=act_by_deg,
                            gamma_sp=(None if gsp is None else float(gsp)),
                            mu=(None if mu is None else float(mu)),
                            gamma=(None if gam is None else float(gam)),
                            cert=(None if cert_val is None
                                  else float(cert_val)),
                            sel=sel,
                            gamma_sp_curve=curve,
                            fit_secs=fit_secs,
                            secs=round(time.time() - tic, 2))
            did = True
            gtxt = f" gsp={gsp:.2e} mu={mu:.2e} sel={sel}" if gsp is not None else ""
            print(f"  r={r:2d} {nm:24s} state(te)={s_te:.3e} out(te)={o_te:.3e} "
                  f"act={res[key]['n_active']:>3d}{gtxt} ({res[key]['secs']}s)")
            _save(res)
    _save(res)
    done = all(f"{nm}@r={r}" in res for r in RS for nm in METHOD_NAMES)
    return res, done


def _save(res):
    PLOTS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(CKPT, "w"), indent=0)


def timeseries(D, res):
    """y(t) = C x for each method vs ground truth over the full horizon."""
    C = D["C"]
    V = {k: D[f"V_{k}"] for k in ("tr", "va", "te")}
    order = ("tr", "va", "te")
    tfull = np.concatenate([D[f"t_{k}"] for k in order])
    Ytrue = np.concatenate([D[f"Y_{k}"] for k in order], axis=1)
    n_out = Ytrue.shape[0]

    CFG["_svd"] = _shared_svd(V["tr"], SVD_CACHE)     # all OPM fits
    # Per-method y(t) cache so the phase is resumable under the 45s limit.
    ycache = PLOTS / "nse_timeseries_cache_full.npz"
    ymethods = dict(np.load(ycache)) if ycache.exists() else {}
    for nm in METHOD_NAMES:
        if nm in ymethods:
            continue
        entry = res.get(f"{nm}@r={R_OUT}") or {}
        model = build_model(nm, V["tr"], R_OUT, CFG,
                            gamma_sp=entry.get("gamma_sp"),
                            mu=entry.get("mu"),
                            sel=entry.get("sel") or "greedy",
                            gamma=entry.get("gamma"))
        yy = [C @ np.asarray(model.reconstruct(V[k])) for k in order]
        ymethods[nm] = np.concatenate(yy, axis=1)
        np.savez(ycache, **ymethods)
        print(f"    cached y(t) for {nm}")

    fig, axes = plt.subplots(3, 2, figsize=(18, 11), sharex=True)
    axes = axes.ravel()
    t_train_end, t_val_end = D["t_tr"][-1], D["t_va"][-1]
    for i in range(n_out):
        ax = axes[i]
        ax.plot(tfull, Ytrue[i], color="k", lw=2.4, label="Ground truth", zorder=10)
        for nm in METHOD_NAMES:
            st = METHOD_STYLES[nm]
            ls = "--" if "--" in st["fmt"] else "-"
            ax.plot(tfull, ymethods[nm][i], ls, color=st["color"],
                    lw=1.4, alpha=0.9, label=nm)
        for tv in (t_train_end, t_val_end):
            ax.axvline(tv, color="gray", lw=1.0, ls=":", zorder=5)
        ax.set_title(f"sensor output $y_{i+1}(t)$", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        if i >= 4:
            ax.set_xlabel("time $t$", fontsize=13)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=13,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("Karman vortex (full data) — sensor outputs $y(t)=Cx$  "
                 f"(r={R_OUT}; dotted lines mark the train/val/test windows)",
                 fontsize=18, y=1.0)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(PLOTS / f"nse_output_timeseries_full.{ext}"), dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print("    saved nse_output_timeseries_full.png/.pdf")


def main(phase="all"):
    D = load()
    print(f"[nse-full] d={D['d']} train {D['V_tr'].shape} val {D['V_va'].shape} "
          f"test {D['V_te'].shape}  n_outputs={D['Y_tr'].shape[0]}  phase={phase}")
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
        timeseries(D, res)
    print("DONE")


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[nse-full] elapsed {time.time()-t0:.1f}s")
