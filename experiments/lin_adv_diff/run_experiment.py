"""Advection-diffusion comparison experiment (driver).

Compares POD, Greedy-QM, Geelen-QM, and the budget-calibrated Greedy-OPM
(degrees {2} and {2, 3, 4}) on the periodic linear advection-diffusion
problem of :mod:`problem`.  Training uses one domain traversal at the
training wave speed; testing uses (i) an unseen wave speed over the same
window and (ii) a contiguous forecast window.  The only Greedy-OPM
hyperparameter is the Lipschitz budget ``L`` (the dilation ``delta`` is
fixed); there is no validation set.

Phases (resumable via the JSON checkpoint; rerun until it prints DONE):

* ``budget`` -- error / active features / certificate vs L at ``R_BUDGET``
  (full refit at every L, so the mode selection is L-consistent).
* ``sweep``  -- all methods for r = 1, ..., R_MAX at the fixed budget.
* ``plots``  -- error-vs-r curves, active-feature bars, budget-study plot.
* ``field``  -- space-time reconstruction / error panels at ``R_FIELD``.
* ``all``    -- everything above, in order.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from methods import (  # noqa: E402
    CFG,
    METHOD_NAMES,
    OPM_DEGREES,
    active_by_degree,
    build_model,
    n_active,
)
from problem import make_dataset  # noqa: E402

PLOTS = _HERE / "plots"
CKPT = PLOTS / "lin_adv_diff_results.json"
CKPT_BUDGET = PLOTS / "lin_adv_diff_budget_study.json"

R_MAX = 10
RS = list(range(1, R_MAX + 1))
R_BUDGET = 8                    # reduced dimension of the L study
R_FIELD = 8                     # reduced dimension of the field panels
L_GRID = [1.5, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 1e4, 1e6]

SPLITS = ("train", "test_param", "test_forecast")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=0))


def _evaluate(model, S: dict) -> dict:
    """Relative reconstruction errors of a fitted model on all splits."""
    return {sp: model.reconstruction_error(S[sp], relative=True)
            for sp in SPLITS}


# ---------------------------------------------------------------------------
# Phase: budget study (error vs L, full refit per L)
# ---------------------------------------------------------------------------


def run_budget_study(S: dict) -> dict:
    """Fit both Greedy-OPM variants at every budget in ``L_GRID``."""
    study = _load(CKPT_BUDGET)
    for name in OPM_DEGREES:
        recs = study.setdefault(name, [])
        done = {rec["L"] for rec in recs}
        for L in L_GRID:
            if L in done:
                continue
            tic = time.time()
            model = build_model(name, S["train"], R_BUDGET, L=L)
            rec = dict(L=L, **_evaluate(model, S),
                       n_active=n_active(model),
                       certificate=model.certificate_,
                       lambda_=model.lambda_,
                       secs=round(time.time() - tic, 1))
            recs.append(rec)
            recs.sort(key=lambda rec: rec["L"])
            _save(CKPT_BUDGET, study)
            print(f"  [budget] {name} r={R_BUDGET} L={L:>8g}: "
                  f"train={rec['train']:.3e} param={rec['test_param']:.3e} "
                  f"forecast={rec['test_forecast']:.3e} "
                  f"act={rec['n_active']} ({rec['secs']}s)")
    return study


# ---------------------------------------------------------------------------
# Phase: r-sweep at the fixed budget
# ---------------------------------------------------------------------------


def run_sweep(S: dict) -> dict:
    """Fit every method at every r; store errors and feature counts."""
    results = _load(CKPT)
    for r in RS:
        for name in METHOD_NAMES:
            key = f"{name}@r={r}"
            if key in results:
                continue
            tic = time.time()
            model = build_model(name, S["train"], r)
            rec = dict(method=name, r=r, **_evaluate(model, S),
                       n_active=n_active(model),
                       n_active_by_degree=active_by_degree(model),
                       certificate=getattr(model, "certificate_", None),
                       lambda_=getattr(model, "lambda_", None),
                       secs=round(time.time() - tic, 1))
            results[key] = rec
            _save(CKPT, results)
            print(f"  r={r:2d}  {name:18s} train={rec['train']:.3e} "
                  f"param={rec['test_param']:.3e} "
                  f"forecast={rec['test_forecast']:.3e} "
                  f"act={rec['n_active']:>3d}  ({rec['secs']}s)")
    return results


# ---------------------------------------------------------------------------
# Phase: plots
# ---------------------------------------------------------------------------


def run_plots() -> None:
    """Render the error-vs-r, active-feature, and budget-study figures."""
    from plots import active_features_bar, budget_study, error_vs_r

    results = _load(CKPT)
    error_vs_r(
        results, RS, METHOD_NAMES,
        str(PLOTS / "lin_adv_diff_error_vs_r"),
        "Advection-diffusion — reconstruction error vs reduced dimension "
        f"(Greedy-OPM at $L = {CFG['L']:g}$, $\\delta = {CFG['delta']:g}$)",
    )
    active_features_bar(
        results, RS, METHOD_NAMES,
        str(PLOTS / "lin_adv_diff_active_features"),
        "Advection-diffusion — active nonlinear features vs reduced dimension",
    )
    study = _load(CKPT_BUDGET)
    if study:
        budget_study(
            study,
            str(PLOTS / "lin_adv_diff_budget_study"),
            "Advection-diffusion — reconstruction error vs Lipschitz budget "
            f"$L$ (full refit per $L$, $r = {R_BUDGET}$)",
        )


# ---------------------------------------------------------------------------
# Phase: field panels (bonus)
# ---------------------------------------------------------------------------


def run_field(data: dict, S: dict) -> None:
    """Space-time reconstruction / error panels on both test splits."""
    from plots import field_panels

    models = {name: build_model(name, S["train"], R_FIELD)
              for name in METHOD_NAMES}
    for split in ("test_param", "test_forecast"):
        truth = np.asarray(data[split]["S"])
        t, x = data[split]["t"], data[split]["x"]
        extent = [t[0], t[-1], x[0], x[-1]]
        recon, rel = {}, {}
        for name, model in models.items():
            recon[name] = np.asarray(model.reconstruct(S[split]))
            rel[name] = model.reconstruction_error(S[split], relative=True)
        tag = ("unseen wave speed" if split == "test_param" else "forecast")
        field_panels(
            truth, recon, METHOD_NAMES,
            str(PLOTS / f"lin_adv_diff_field_{split}"),
            f"Advection-diffusion — space-time reconstruction "
            f"({tag}, $r = {R_FIELD}$)",
            extent=extent, rel_errs=rel,
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(phase: str = "all") -> None:
    data = make_dataset()
    S = {sp: jnp.asarray(data[sp]["S"]) for sp in SPLITS}
    print(f"[lin_adv_diff] d={S['train'].shape[0]} "
          f"n_train={S['train'].shape[1]} "
          f"n_test={S['test_param'].shape[1]}/{S['test_forecast'].shape[1]} "
          f"L={CFG['L']:g} delta={CFG['delta']:g} phase={phase}")

    if phase in ("all", "budget"):
        run_budget_study(S)
    if phase in ("all", "sweep"):
        run_sweep(S)
    if phase in ("all", "plots"):
        run_plots()
    if phase in ("all", "field"):
        run_field(data, S)
    print("DONE")


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[lin_adv_diff] elapsed {time.time() - t0:.1f}s")
