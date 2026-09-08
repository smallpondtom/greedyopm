"""Grid-adaptive, per-variant Lipschitz-budget selection.

Selects `L*` for each Greedy-OPM variant automatically from the validation
splits -- nothing is hard-coded in ``methods.py``.  Two things make it work:

1. **Two-regime validation objective.**  The budget must serve both an
   *interpolated* test and an *extrapolated* test, so the objective is the
   geometric mean of the two validation relative errors,
   ``sqrt(e_interp * e_extrap)``.  The geometric mean balances the two on a
   log scale (they differ by orders of magnitude): a large budget wins on
   interpolation but blows up on extrapolation, a small budget is the
   reverse, and the minimiser is the stable compromise.

2. **Grid-adaptive refinement to 0.01.**  The sweep starts on a coarse
   log grid over a wide range, brackets the minimiser by its neighbours,
   and repeats on a finer *linear* grid inside that bracket -- both the
   swept range and the interval shrink each stage -- until the grid step
   falls to ``PRECISION = 0.01`` (second-decimal precision; refining
   further would just fit validation noise).

The sweep is evaluated cheaply on a single reference fit per variant via
``GreedyOPM.set_budget`` (recalibrates the lifting for a new budget without
re-selecting modes).  To keep the modes consistent with the chosen budget
the reference is refit once at the first-pass optimum (a one-step fixed
point in `L`); the final r-sweep then refits fully at ``L*``.
"""

from __future__ import annotations

import numpy as np

from methods import build_model

PRECISION = 0.01          # stop when the grid step reaches this (2 decimals)
L_LO = 1.05               # smallest admissible budget (L > 1)
L_HI = 1.0e5              # largest budget of the initial coarse sweep
N_PTS = 15                # grid points per stage
MAX_STAGES = 30           # safety cap on refinement stages
L_REF0 = 100.0            # starting budget for the reference fit (pre-iterate)


def val_objective(pm, S) -> tuple:
    """Geometric mean of the interp / extrap validation relative errors."""
    ei = float(pm.reconstruction_error(S["val_interp"], relative=True))
    ee = float(pm.reconstruction_error(S["val_extrap"], relative=True))
    return float(np.sqrt(max(ei, 1e-300) * max(ee, 1e-300))), ei, ee


def budget_curve(pm, S, grid, splits) -> list:
    """Evaluate every split (+ active count / certificate) over ``grid``."""
    opm = pm.model
    recs = []
    for L in grid:
        opm.set_budget(float(L))
        rec = dict(
            L=float(L),
            **{sp: float(pm.reconstruction_error(S[sp], relative=True))
               for sp in splits},
            objective=val_objective(pm, S)[0],
            n_active=int(getattr(pm, "n_active_nonlinear", 0)),
            certificate=float(getattr(pm, "certificate_", np.nan)),
        )
        recs.append(rec)
    return recs


def _adaptive(pm, S, log=print) -> tuple:
    """Return ``(L_star, history)`` by progressive grid narrowing."""
    opm = pm.model
    hist = []

    def obj_at(L):
        L = round(float(L), 2)
        opm.set_budget(L)
        o, ei, ee = val_objective(pm, S)
        hist.append(dict(L=L, objective=o, val_interp=ei, val_extrap=ee,
                         n_active=int(pm.n_active_nonlinear)))
        return o

    lo, hi = float(L_LO), float(L_HI)
    best_L, best_obj = round(float(L_LO), 2), np.inf
    for stage in range(MAX_STAGES):
        grid = (np.geomspace(lo, hi, N_PTS) if stage == 0
                else np.linspace(lo, hi, N_PTS))
        objs = [obj_at(L) for L in grid]
        i = int(np.argmin(objs))
        if objs[i] < best_obj:
            best_obj, best_L = objs[i], round(float(grid[i]), 2)
        lo_new = float(grid[max(i - 1, 0)])
        hi_new = float(grid[min(i + 1, N_PTS - 1)])
        step = (hi_new - lo_new) / (N_PTS - 1)
        log(f"    [Lsweep] stage {stage:2d}: L*={best_L:<10g} "
            f"obj={best_obj:.3e} bracket=[{lo_new:.4g}, {hi_new:.4g}] "
            f"step={step:.3g}")
        if step <= PRECISION or (hi_new - lo_new) <= PRECISION:
            break
        lo, hi = lo_new, hi_new
    return best_L, hist


def select_budget(name, S, plot_grid, splits, *, r_budget, log=print) -> dict:
    """Reference-fit, one-step-refit, and grid-adaptive select for a variant.

    Returns a study record with the full-range ``curve`` (for plotting), the
    refinement ``history``, and the selected ``L_star``.
    """
    # First pass: fit at a moderate budget, pick a preliminary L.
    pm = build_model(name, S["train"], r_budget, L=L_REF0)
    L1, _ = _adaptive(pm, S, log=lambda *_: None)
    log(f"  [{name}] first-pass L ~ {L1:g}; refitting modes there", flush=True)
    # Second pass: refit modes at the preliminary L (one-step fixed point),
    # then do the reported adaptive sweep and full-range curve there.
    pm = build_model(name, S["train"], r_budget, L=max(L1, L_LO))
    L_star, hist = _adaptive(pm, S, log=log)
    curve = budget_curve(pm, S, plot_grid, splits)
    log(f"  [{name}] selected L* = {L_star:g} "
        f"(objective {min(h['objective'] for h in hist):.3e})", flush=True)
    return dict(L_star=float(L_star), curve=curve, history=hist)
