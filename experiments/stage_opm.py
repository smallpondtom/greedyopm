"""Staged fitting of expensive Greedy-OPM entries (degree-4 dictionaries).

The quartic dictionary makes a single greedy fit exceed the per-call
wall-clock limit at larger r, so this driver spreads ONE (method, r)
checkpoint entry across calls using the resumable greedy loop of
``GreedyOPM.fit`` (``J_init`` / ``max_new_iters``): each call performs a
bounded number of greedy iterations (or the finalize + tuning step) for one
mode-selection variant, checkpointing the loop state (the selected-mode
list J) to JSON.  When both variants are tuned, the winning variant's entry
is written into the experiment's results checkpoint in the exact schema of
the experiment's own sweep.

Usage:  python stage_opm.py <swe|nse|nsefull> <r> [chunk_iters]
Repeat until it prints "ENTRY DONE".
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "shallow_water"))
sys.path.insert(0, str(_HERE / "data"))
sys.path.insert(0, str(_HERE / "nse"))
sys.path.insert(0, str(_HERE.parent / "src"))

from greedyopm import GreedyOPM                                # noqa: E402
from exp_methods import (_grid_curves, _select_config,         # noqa: E402
                         _select_variant, _apply_opm_params,
                         _active_by_degree)
from exp_split import split_indices                            # noqa: E402

METHOD = "Greedy-OPM (p=[2,3,4])"
DEGREES = [2, 3, 4]


def _make_model(cfg, r, s_pool, svd_unused):
    budget = cfg.get("budget")
    if budget is not None and not np.isfinite(budget):
        budget = None
    return GreedyOPM(
        r=r, s=s_pool, p=DEGREES, gamma_sp=0.0, gamma=cfg["gamma_opm"],
        ortho_method="arnoldi", center=True, n_reorth=2, budget=budget,
        grad_dilation=cfg.get("grad_dilation", 0.1),
        n_grad_samples=cfg.get("n_grad_samples"))


def _load(exp):
    """Return (cfg, ckpt_path, S_tr, S_va, svd, metrics_fn, entry_fn)."""
    if exp == "swe":
        from shallow_water_compare import load_height, CFG, CKPT
        S, _ = load_height()
        itr, iva, ite = split_indices(S.shape[1])
        S_tr = jnp.asarray(S[:, itr]); S_va = jnp.asarray(S[:, iva])
        S_te = jnp.asarray(S[:, ite])

        def metrics(m):
            return dict(train=m.reconstruction_error(S_tr),
                        val=m.reconstruction_error(S_va),
                        test=m.reconstruction_error(S_te))

        def entry(r, best, st):
            return dict(method=METHOD, r=int(r), **st["metrics"],
                        n_active=st["n_active"],
                        n_active_by_degree=st["by_deg"],
                        gamma_sp=float(best["gamma_sp"]),
                        gamma=float(best["gamma"]), cert=st["cert"],
                        sel=st["sel"], gamma_sp_curve=st["curve"],
                        fit_secs=st["secs"], secs=st["secs"])
        return dict(CFG), Path(CKPT), S_tr, S_va, None, metrics, entry

    if exp == "nse":
        from nse_compare import load, CFG, CKPT, _metrics
        D = load()
        S_tr = jnp.asarray(D["A_train"]); S_va = jnp.asarray(D["A_val"])
        splits = {t: (D[f"A_{t}"], D[f"Y_{t}"], float(D[f"floor_{t}"]),
                      float(D[f"nrm_{t}"]))
                  for t in ("train", "val", "test")}

        def metrics(m):
            out = {}
            for tag, (A, Y, floor, nrm) in splits.items():
                s_rel, o_rel = _metrics(m, A, Y, floor, nrm,
                                        D["C_red"], D["y_ref"])
                out[f"state_{tag}"] = s_rel
                out[f"out_{tag}"] = o_rel
            return out

        def entry(r, best, st):
            return dict(method=METHOD, r=int(r), **st["metrics"],
                        n_active=st["n_active"],
                        n_active_by_degree=st["by_deg"],
                        gamma_sp=float(best["gamma_sp"]),
                        gamma=float(best["gamma"]), cert=st["cert"],
                        sel=st["sel"], gamma_sp_curve=st["curve"],
                        fit_secs=st["secs"], secs=st["secs"])
        return dict(CFG), Path(CKPT), S_tr, S_va, None, metrics, entry

    if exp == "nsefull":
        from nse_compare_full import (load, CFG, CKPT, _metrics,
                                      _shared_svd, SVD_CACHE)
        D = load()
        svd = _shared_svd(D["V_tr"], SVD_CACHE)
        S_tr, S_va = D["V_tr"], D["V_va"]
        C = D["C"]

        def metrics(m):
            out = {}
            for tag, key in (("train", "tr"), ("val", "va"), ("test", "te")):
                s_rel, o_rel = _metrics(m, D[f"V_{key}"], D[f"Y_{key}"],
                                        D["nrm"][tag], C)
                out[f"state_{tag}"] = s_rel
                out[f"out_{tag}"] = o_rel
            return out

        def entry(r, best, st):
            return dict(method=METHOD, r=int(r), **st["metrics"],
                        n_active=st["n_active"],
                        n_active_by_degree=st["by_deg"],
                        gamma_sp=float(best["gamma_sp"]),
                        gamma=float(best["gamma"]), cert=st["cert"],
                        sel=st["sel"], gamma_sp_curve=st["curve"],
                        fit_secs=st["secs"], secs=st["secs"])
        return dict(CFG), Path(CKPT), S_tr, S_va, svd, metrics, entry

    raise SystemExit(f"unknown experiment {exp!r}")


def main(exp, r, chunk=4):
    cfg, ckpt, S_tr, S_va, svd, metrics, entry_fn = _load(exp)
    key = f"{METHOD}@r={r}"
    res = json.load(open(ckpt)) if ckpt.exists() else {}
    if key in res:
        print("ENTRY DONE (already present)")
        return
    state_p = ckpt.parent / f"__p4stage_{exp}_r{r}.json"
    st = json.load(open(state_p)) if state_p.exists() else {}

    pool = {"greedy": cfg.get("s_pool_p4", 24), "leading": r}
    for sel in ("greedy", "leading"):
        s = st.get(sel, {})
        if s.get("done"):
            continue
        tic = time.time()
        m = _make_model(cfg, r, pool[sel], svd)
        m.fit(S_tr, svd=svd, J_init=s.get("J"), max_new_iters=chunk)
        if m.partial_J_ is not None:
            st[sel] = dict(J=[int(j) for j in m.partial_J_], done=False,
                           secs=s.get("secs", 0.0) + time.time() - tic)
            json.dump(st, open(state_p, "w"))
            print(f"[stage] {exp} r={r} {sel}: "
                  f"{len(m.partial_J_)}/{r} modes; rerun to continue")
            return
        curves = _grid_curves(m, S_tr, S_va, cfg,
                              cfg.get("gamma_sp_points", 24), sel)
        best = _select_config(curves, cfg.get("gamma_sp_tol", 0.02))
        _apply_opm_params(m, best["gamma_sp"], gamma=best.get("gamma"))
        st[sel] = dict(
            J=s.get("J"), done=True, best=best,
            curve=next(c for c in curves
                       if c and c[0].get("gamma") == best.get("gamma")),
            n_active=int(m.n_selected_nonlinear),
            by_deg=_active_by_degree(m), cert=float(m.certificate_),
            sel=sel, metrics=metrics(m),
            secs=round(s.get("secs", 0.0) + time.time() - tic, 2))
        json.dump(st, open(state_p, "w"))
        print(f"[stage] {exp} r={r} {sel}: tuned "
              f"(val={best['val_err']:.4f}); rerun for next stage")
        return

    stages = {sel: (None, None, st[sel]["best"])
              for sel in ("greedy", "leading")}
    best_sel = _select_variant(stages, cfg.get("variant_tol",
                                               cfg.get("gamma_sp_tol", 0.02)))
    s = st[best_sel]
    res[key] = entry_fn(r, s["best"], s)
    json.dump(res, open(ckpt, "w"), indent=0)
    state_p.unlink(missing_ok=True)
    mkeys = [k for k in s["metrics"] if "test" in k or k == "test"]
    print(f"[stage] {exp} r={r}: wrote entry sel={best_sel} "
          + " ".join(f"{k}={s['metrics'][k]:.4f}" for k in mkeys))
    print("ENTRY DONE")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]),
         int(sys.argv[3]) if len(sys.argv) > 3 else 4)
