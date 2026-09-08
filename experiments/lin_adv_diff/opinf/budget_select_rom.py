"""Lipschitz-budget selection against the reduced-model error (note, Sec. 6).

The reconstruction-selected budgets of the manifold experiment are loose --
appropriate for encode-decode accuracy, but past the certified collar a loose
budget "returns the plain model's divergence" (``notes/opinf_opm.tex``): the
GMG mass matrix ``M = I + J_p^T Vbar^T Vbar J_p`` is only conditioned by the
budget (``cond(M) <= L^2``), so carrying *dynamics* requires an ``L`` chosen for
the reduced model.  Per the note, the budget of each Greedy-OPM variant is
therefore selected by minimizing the **reduced-model** error on the held-out
*validation* speeds (spline-interpolated operators; the test speed is never
touched), with diverged rollouts penalized.

The sweep is evaluated through ``set_budget`` (closed-form recalibration of the
lifting -- no refit per grid point) on a reference fit, with the same two-pass
scheme as the reconstruction budget selection: a first pass from a moderate
reference budget locates the optimum, the manifold is refit there (mode
selection consistent with the budget), and the reported sweep runs on the
refitted model.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from opinf_methods import ManifoldROM
from greedyopm import opinf

L_LO, L_HI = 1.5, 5.0e3
N_COARSE, N_REFINE = 12, 8
L_REF0 = 100.0            # reference-fit budget for the first pass
PENALTY = 1.0e3           # error cap: a diverged rollout scores at the cap


def _rom_val_error(mrom: ManifoldROM, data: dict, kind: str = "gmg") -> float:
    """Geometric-mean ROM error over the held-out validation speeds."""
    import jax
    jax.clear_caches()  # every eval compiles fresh kernels; cap JIT memory
    dt = data["dt"]
    from run_opinf import _learn_all, _ops_for
    learned = _learn_all(mrom, data, kind)
    errs = []
    for c_v, S_v in zip(data["val"]["speeds"], data["val"]["blocks"]):
        ops_v = _ops_for(mrom, learned, data, c_v, kind)
        e = opinf.relative_state_error(S_v, mrom.simulate(ops_v, S_v, dt, kind))
        errs.append(min(e, PENALTY) if np.isfinite(e) else PENALTY)
    return float(np.exp(np.mean(np.log(np.maximum(errs, 1e-300)))))


def _sweep(mrom: ManifoldROM, data: dict, grid, kind: str, log) -> List[dict]:
    recs = []
    for L in grid:
        mrom.set_budget(float(L))
        obj = _rom_val_error(mrom, data, kind)
        recs.append(dict(L=float(L), objective=obj,
                         n_active=int(mrom.n_features)))
        log(f"      L={L:10.4g}  val ROM err={obj:.3e}  "
            f"act={mrom.n_features}", flush=True)
    return recs


def select_budget_rom(name: str, r_budget: int, data: dict, *,
                      kind: str = "gmg", log=print) -> Dict:
    """Two-pass grid selection of ``L*`` minimizing the validation ROM error."""
    grid = np.geomspace(L_LO, L_HI, N_COARSE)

    log(f"  [{name}] ROM budget pass 1 (reference fit at L={L_REF0:g})",
        flush=True)
    mrom = ManifoldROM(name, r_budget, data["train"]["S_fit"], L=L_REF0)
    recs1 = _sweep(mrom, data, grid, kind, log)
    i = int(np.argmin([rec["objective"] for rec in recs1]))
    L1 = recs1[i]["L"]

    log(f"  [{name}] ROM budget pass 2 (refit at L={L1:g})", flush=True)
    mrom = ManifoldROM(name, r_budget, data["train"]["S_fit"], L=L1)
    recs2 = _sweep(mrom, data, grid, kind, log)
    i = int(np.argmin([rec["objective"] for rec in recs2]))
    L_star, best = recs2[i]["L"], recs2[i]["objective"]
    lo = recs2[max(i - 1, 0)]["L"]
    hi = recs2[min(i + 1, len(recs2) - 1)]["L"]
    recs3 = _sweep(mrom, data, np.linspace(lo, hi, N_REFINE), kind, log)
    for rec in recs3:
        if rec["objective"] < best:
            best, L_star = rec["objective"], rec["L"]

    log(f"  [{name}] selected ROM budget L* = {L_star:g} "
        f"(val ROM err {best:.3e})", flush=True)
    return dict(L_star=float(L_star), best_objective=float(best),
                L_refit=float(L1),
                curve=sorted(recs2 + recs3, key=lambda d: d["L"]))


# ---------------------------------------------------------------------------
# Per-r budget selection (compact single-pass sweep around the r=10 anchor)
# ---------------------------------------------------------------------------

GRID_PER_R = np.geomspace(1.5, 100.0, 6)

#: Reference-fit budgets: the greedy mode selection depends on the budget the
#: manifold is *fit* at, and a tight reference recalibrated upward via
#: ``set_budget`` often lands in a different (better) stability basin than a
#: direct fit.  The selection therefore searches (L_ref, L) pairs and the
#: deployed model is constructed exactly the same way: fit at ``L_ref``, then
#: ``set_budget(L)``.
REF_LS = (1.5, 100.0)


def select_budget_rom_per_r(name: str, rs, data: dict, anchor_L: float, *,
                            kind: str = "gmg", log=print, done=None,
                            checkpoint_cb=None, extra_refs=()) -> Dict:
    """Select ``(L_ref, L)*(r)`` for every reduced dimension.

    ROM stability is a per-``r`` property, so the note's reduced-model budget
    selection is applied per ``r``: for each reference budget in ``REF_LS``
    (plus the r=10 anchor) the manifold is fit once and the budget swept through
    the closed-form ``set_budget`` recalibration; the pair minimizing the
    geometric-mean validation ROM error wins.  The final sweep deploys the
    selected pair verbatim (fit at ``L_ref``, recalibrate to ``L``).
    """
    refs = sorted(set([float(x) for x in REF_LS] + [float(anchor_L)]
                      + [float(x) for x in extra_refs]))
    grid = np.unique(np.concatenate([GRID_PER_R, [float(anchor_L)]]))
    per_r, curves = dict(done or {}), {}
    for r in rs:
        if int(r) in per_r or str(r) in per_r:
            per_r[int(r)] = dict(per_r.pop(str(r), per_r.get(int(r))))
            log(f"    [{name}] r={r:2d}: cached", flush=True)
            continue
        best = None
        recs = []
        for L_ref in refs:
            mrom = ManifoldROM(name, int(r), data["train"]["S_fit"], L=L_ref)
            for L in grid:
                mrom.set_budget(float(L))
                obj = _rom_val_error(mrom, data, kind)
                rec = dict(L_ref=float(L_ref), L=float(L), objective=obj,
                           n_active=int(mrom.n_features))
                recs.append(rec)
                if best is None or obj < best["objective"]:
                    best = rec
        per_r[int(r)] = dict(L_ref=best["L_ref"], L=best["L"])
        curves[int(r)] = recs
        log(f"    [{name}] r={r:2d}: L_ref={best['L_ref']:g} "
            f"L*(r)={best['L']:.4g} val ROM err={best['objective']:.3e}",
            flush=True)
        if checkpoint_cb is not None:
            checkpoint_cb(dict(per_r=per_r, curves=curves,
                               anchor_L=float(anchor_L)))
    return dict(per_r=per_r, curves=curves, anchor_L=float(anchor_L))
