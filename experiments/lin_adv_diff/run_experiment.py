"""Advection-diffusion comparison experiment (driver).

Compares POD, Greedy-QM, Geelen-QM, and the budget-calibrated Greedy-OPM
(degrees {2} and {2, 3}) on the periodic linear advection-diffusion problem
of :mod:`problem`, now over a **wide** range of wave speeds and with **two**
test scenarios:

* ``test_interp`` -- a random interior wave speed (interpolation);
* ``test_extrap`` -- a wave speed beyond the training range (extrapolation),
  where an unregularised polynomial manifold extrapolates wildly and the
  Lipschitz budget earns its keep.

The Lipschitz budget ``L`` is chosen **automatically and per variant**
(``p=2`` and ``p=3`` separately) by the grid-adaptive sweep in
:mod:`budget_select`: a geometric-mean objective over the interpolation and
extrapolation *validation* splits, refined to 0.01 precision.  Nothing is
hard-coded in ``methods.py``.

Phases (resumable via the JSON checkpoints):

* ``budget`` -- select ``L*_{p2}`` and ``L*_{p3}`` (grid-adaptive), and
  record the full-range budget curve for the figures.
* ``sweep``  -- all methods for r = 1..R_MAX (each Greedy-OPM at its own L*).
* ``plots``  -- error-vs-r curves, active-feature bars, budget-study plot.
* ``field``  -- space-time reconstruction / error panels (interp + extrap).
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
from problem import SPLITS, make_dataset  # noqa: E402
from budget_select import select_budget  # noqa: E402

PLOTS = _HERE / "plots"
CKPT = PLOTS / "lin_adv_diff_results.json"
CKPT_BUDGET = PLOTS / "lin_adv_diff_budget_study.json"
REPORT = PLOTS / "lin_adv_diff_report.json"

R_MAX = 10
RS = list(range(1, R_MAX + 1))
R_BUDGET = 10                    # reduced dimension of the L selection
R_FIELD = 10                     # reduced dimension of the field panels

#: Full-range grid on which the budget curve is drawn (cheap, set_budget).
PLOT_GRID = list(np.geomspace(1.05, 1.0e5, 30))
#: Splits whose error is recorded along the budget curve.
CURVE_SPLITS = ("train", "val_interp", "val_extrap", "test_interp",
                "test_extrap")

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
# Phase: grid-adaptive, per-variant budget selection
# ---------------------------------------------------------------------------


def run_budget_selection(S: dict) -> dict:
    """Select ``L*`` for each Greedy-OPM variant; record the budget curves."""
    study = _load(CKPT_BUDGET)
    for name in OPM_DEGREES:
        if name in study and "L_star" in study[name]:
            print(f"  [budget] {name}: cached L* = {study[name]['L_star']:g}",
                  flush=True)
            continue
        tic = time.time()
        rec = select_budget(name, S, PLOT_GRID, CURVE_SPLITS,
                            r_budget=R_BUDGET, log=print)
        rec["secs"] = round(time.time() - tic, 1)
        study[name] = rec
        _save(CKPT_BUDGET, study)
    return study


def resolve_Ls() -> dict:
    """Selected budgets ``{variant: L*}`` (raises if selection not run)."""
    study = _load(CKPT_BUDGET)
    Ls = {name: study[name]["L_star"] for name in OPM_DEGREES
          if name in study and "L_star" in study[name]}
    if len(Ls) != len(OPM_DEGREES):
        raise RuntimeError("Budget not selected yet; run the 'budget' phase.")
    return Ls


# ---------------------------------------------------------------------------
# Phase: r-sweep at the selected budgets
# ---------------------------------------------------------------------------


def run_sweep(S: dict, Ls: dict) -> dict:
    """Fit every method at every r; each Greedy-OPM variant uses its own L*."""
    results = _load(CKPT)
    results.setdefault("_meta", {})["L_star"] = Ls
    for r in RS:
        for name in METHOD_NAMES:
            key = f"{name}@r={r}"
            if key in results:
                continue
            tic = time.time()
            model = build_model(name, S["train"], r, L=Ls.get(name))
            rec = dict(method=name, r=r, **_evaluate(model, S),
                       n_active=n_active(model),
                       n_active_by_degree=active_by_degree(model),
                       certificate=getattr(model, "certificate_", None),
                       lambda_=getattr(model, "lambda_", None),
                       secs=round(time.time() - tic, 1))
            results[key] = rec
            _save(CKPT, results)
            print(f"  r={r:2d}  {name:18s} "
                  f"train={rec['train']:.2e} "
                  f"t_in={rec['test_interp']:.2e} "
                  f"t_ex={rec['test_extrap']:.2e} "
                  f"act={rec['n_active']:>3d}  ({rec['secs']}s)", flush=True)
    return results


# ---------------------------------------------------------------------------
# Phase: plots
# ---------------------------------------------------------------------------


def run_plots(Ls: dict) -> None:
    """Render every figure (journal-quality, no titles) from the checkpoints."""
    from plots import (active_features_bar, budget_study,
                       certificate_vs_budget, error_vs_r)

    results = _load(CKPT)
    error_vs_r(results, RS, METHOD_NAMES,
               str(PLOTS / "lin_adv_diff_error_vs_r"),
               splits=("train", "test_interp", "test_extrap"))
    active_features_bar(results, RS, METHOD_NAMES,
                        str(PLOTS / "lin_adv_diff_active_features"))
    certificate_vs_budget(results, RS, list(OPM_DEGREES), Ls,
                          str(PLOTS / "lin_adv_diff_certificate"))
    study = _load(CKPT_BUDGET)
    if study:
        budget_study(study, str(PLOTS / "lin_adv_diff_budget_study"))


# ---------------------------------------------------------------------------
# Phase: field panels (interp + extrap)
# ---------------------------------------------------------------------------


def run_field(data: dict, S: dict, Ls: dict) -> None:
    """Space-time reconstruction / error panels on both test splits."""
    from plots import field_panels

    models = {name: build_model(name, S["train"], R_FIELD, L=Ls.get(name))
              for name in METHOD_NAMES}
    for split, tag in (("test_interp", "interpolated"),
                       ("test_extrap", "extrapolated")):
        truth = np.asarray(data[split]["S"])
        t, x = data[split]["t"], data[split]["x"]
        extent = [t[0], t[-1], x[0], x[-1]]
        recon, rel = {}, {}
        for name, model in models.items():
            recon[name] = np.asarray(model.reconstruct(S[split]))
            rel[name] = model.reconstruction_error(S[split], relative=True)
        field_panels(
            truth, recon, METHOD_NAMES,
            str(PLOTS / f"lin_adv_diff_field_{split}"),
            extent=extent, rel_errs=rel,
        )


# ---------------------------------------------------------------------------
# Phase: journal report (values worth reporting in a paper)
# ---------------------------------------------------------------------------


def run_report(data: dict, Ls: dict) -> dict:
    """Collect the journal-reportable quantities into a single JSON."""
    import problem as prob

    results = _load(CKPT)
    study = _load(CKPT_BUDGET)

    # certificate validity across r (primal-dual check)
    certificate_validity = {}
    for name in OPM_DEGREES:
        L = float(Ls[name])
        per_r, ratios = [], []
        for r in RS:
            c = results[f"{name}@r={r}"].get("certificate")
            ratio = None if c is None else c / L
            if ratio is not None:
                ratios.append(ratio)
            per_r.append(dict(r=r, certificate=c, budget=L, ratio=ratio,
                              valid=(c is not None and c <= L * (1 + 1e-9))))
        certificate_validity[name] = dict(
            budget=L, per_r=per_r,
            max_ratio=(max(ratios) if ratios else None),
            all_valid=all(x["valid"] for x in per_r),
        )

    # per-method training times, errors, active features
    training_times_s, errors, active_features = {}, {}, {}
    for name in METHOD_NAMES:
        training_times_s[name] = {
            str(r): results[f"{name}@r={r}"]["secs"] for r in RS}
        errors[name] = {
            str(r): {sp: results[f"{name}@r={r}"][sp] for sp in SPLITS}
            for r in RS}
        active_features[name] = {
            str(r): results[f"{name}@r={r}"]["n_active"] for r in RS}

    budget_selection = {
        name: dict(L_star=float(study[name]["L_star"]),
                   selection_time_s=study[name].get("secs"),
                   n_evaluations=len(study[name].get("history", [])),
                   best_objective=min((h["objective"]
                                       for h in study[name].get("history", [])),
                                      default=None))
        for name in OPM_DEGREES if name in study
    }

    report = dict(
        experiment="lin_adv_diff",
        problem="periodic linear advection-diffusion",
        parameter="wave speed c",
        config=dict(
            state_dim=int(data["train"]["S"].shape[0]),
            n_train=int(data["train"]["S"].shape[1]),
            n_params_train=len(data["train"]["speeds"]),
            snaps_per_param=data["train"]["n_per"],
            train_range=[prob.C_MIN, prob.C_MAX],
            train_speeds=data["train"]["speeds"],
            val_interp_speeds=data["val_interp"]["speeds"],
            val_extrap_speed=data["val_extrap"]["c"],
            test_interp_speed=data["test_interp"]["c"],
            test_extrap_speed=data["test_extrap"]["c"],
            nu=prob.NU, delta=CFG["delta"], s_pool=CFG["s_pool"],
            n_grad_samples=CFG["n_grad_samples"], r_max=R_MAX,
            r_budget=R_BUDGET,
        ),
        selected_budgets=Ls,
        budget_selection=budget_selection,
        certificate_validity=certificate_validity,
        training_times_s=training_times_s,
        errors=errors,
        active_features=active_features,
    )
    _save(REPORT, report)
    ok = all(v["all_valid"] for v in certificate_validity.values())
    print(f"    saved {REPORT.name}  (certificate <= budget for all r: {ok})",
          flush=True)
    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(phase: str = "all") -> None:
    data = make_dataset()
    S = {sp: jnp.asarray(data[sp]["S"]) for sp in SPLITS}
    print(f"[lin_adv_diff] d={S['train'].shape[0]} "
          f"n_train={S['train'].shape[1]} "
          f"({len(data['train']['speeds'])} speeds x {data['train']['n_per']}) "
          f"delta={CFG['delta']:g} phase={phase}", flush=True)
    print(f"  train speeds:  {np.round(data['train']['speeds'], 3).tolist()}",
          flush=True)
    print(f"  val_interp:    {np.round(data['val_interp']['speeds'], 3).tolist()}"
          f"   val_extrap: {data['val_extrap']['c']:.3f}", flush=True)
    print(f"  test_interp:   {data['test_interp']['c']:.3f}"
          f"   test_extrap: {data['test_extrap']['c']:.3f}", flush=True)

    if phase in ("all", "budget"):
        run_budget_selection(S)
    Ls = resolve_Ls()
    print(f"[lin_adv_diff] selected budgets: "
          + ", ".join(f"{k} -> L*={v:g}" for k, v in Ls.items()), flush=True)
    if phase in ("all", "sweep"):
        run_sweep(S, Ls)
    if phase in ("all", "plots"):
        run_plots(Ls)
    if phase in ("all", "field"):
        run_field(data, S, Ls)
    if phase in ("all", "plots", "report"):
        run_report(data, Ls)
    print("DONE")


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[lin_adv_diff] elapsed {time.time() - t0:.1f}s")
