"""NACA 0012 parametric (angle-of-attack) comparison experiment (driver).

Compares POD, Greedy-QM, Greedy-CM, Geelen-QM, and the budget-calibrated
Greedy-OPM (degrees {2} and {2, 3}) on the parametric NACA 0012 problem of
:mod:`problem` (laminar SU2 data at Re_c = 1000, see ``run_param.py``), over a
range of angles of attack and with **two** test scenarios:

* ``test_interp`` -- a held-out interior angle (interpolation);
* ``test_extrap`` -- a deep-stall angle beyond the training range
  (extrapolation), where a loose (large-``L``) polynomial manifold
  over-extrapolates and the Lipschitz budget earns its keep.

The Lipschitz budget ``L`` is chosen **automatically and per variant** by the
grid-adaptive sweep in :mod:`budget_select`: a geometric-mean objective over the
interpolation and extrapolation *validation* splits, refined to 0.01 precision.

Phases (resumable via the JSON checkpoints): ``budget``, ``sweep``, ``plots``,
``field``, ``report``, ``all``.  Usage:  python run_experiment.py [phase]
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
    CFG, METHOD_NAMES, OPM_DEGREES, active_by_degree, build_model, n_active,
)
from problem import SPLITS, make_dataset, speed  # noqa: E402
import problem as prob  # noqa: E402
from budget_select import select_budget  # noqa: E402

PLOTS = _HERE / "plots"
CKPT = PLOTS / "naca_results.json"
CKPT_BUDGET = PLOTS / "naca_budget_study.json"
REPORT = PLOTS / "naca_report.json"

R_MAX = 10
RS = list(range(1, R_MAX + 1))
R_BUDGET = 10
R_FIELD = 10

PLOT_GRID = list(np.geomspace(1.05, 1.0e5, 30))
CURVE_SPLITS = ("train", "val_interp", "val_extrap", "test_interp",
                "test_extrap")


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=0))


def _evaluate(model, S: dict) -> dict:
    return {sp: model.reconstruction_error(S[sp], relative=True)
            for sp in SPLITS}


def run_budget_selection(S: dict) -> dict:
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
    study = _load(CKPT_BUDGET)
    Ls = {name: study[name]["L_star"] for name in OPM_DEGREES
          if name in study and "L_star" in study[name]}
    if len(Ls) != len(OPM_DEGREES):
        raise RuntimeError("Budget not selected yet; run the 'budget' phase.")
    return Ls


def run_sweep(S: dict, Ls: dict) -> dict:
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


def run_plots(Ls: dict) -> None:
    from plots import (active_features_bar, budget_study,
                       certificate_vs_budget, error_vs_r)
    results = _load(CKPT)
    if results:
        error_vs_r(results, RS, METHOD_NAMES, str(PLOTS / "naca_error_vs_r"),
                   splits=("train", "test_interp", "test_extrap"))
        active_features_bar(results, RS, METHOD_NAMES,
                            str(PLOTS / "naca_active_features"))
        certificate_vs_budget(results, RS, list(OPM_DEGREES), Ls,
                              str(PLOTS / "naca_certificate"))
    study = _load(CKPT_BUDGET)
    if study:
        budget_study(study, str(PLOTS / "naca_budget_study"))


def run_field(data: dict, S: dict, Ls: dict) -> None:
    from plots import field_panels
    g = prob.load_geom()
    models = {name: build_model(name, S["train"], R_FIELD, L=Ls.get(name))
              for name in METHOD_NAMES}
    for split, tag in (("test_interp", "interp"), ("test_extrap", "extrap")):
        Sfull = S[split]
        pod = np.asarray(models["POD"].reconstruct(Sfull))
        truth = np.asarray(Sfull)
        per_col = (np.linalg.norm(pod - truth, axis=0)
                   / np.linalg.norm(truth, axis=0))
        col = int(np.argmax(per_col))
        truth_col = np.asarray(data[split]["S"][:, col])
        recon, rel = {}, {}
        for name, model in models.items():
            approx = np.asarray(model.reconstruct(Sfull[:, col:col + 1]))[:, 0]
            recon[name] = speed(approx)
            rel[name] = float(model.reconstruction_error(Sfull, relative=True))
        alpha = data[split]["alpha"][0]
        field_panels(
            speed(truth_col), recon, METHOD_NAMES,
            str(PLOTS / f"naca_field_{tag}"),
            geom=g, rel_errs=rel,
            title=rf"$\alpha={alpha:g}^\circ$ ({tag}olation test)",
        )


def run_report(data: dict, Ls: dict) -> dict:
    results = _load(CKPT)
    study = _load(CKPT_BUDGET)

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
            all_valid=all(x["valid"] for x in per_r))

    training_times_s, errors, active_features = {}, {}, {}
    for name in METHOD_NAMES:
        training_times_s[name] = {str(r): results[f"{name}@r={r}"]["secs"]
                                  for r in RS}
        errors[name] = {str(r): {sp: results[f"{name}@r={r}"][sp]
                                 for sp in SPLITS} for r in RS}
        active_features[name] = {str(r): results[f"{name}@r={r}"]["n_active"]
                                 for r in RS}

    budget_selection = {
        name: dict(L_star=float(study[name]["L_star"]),
                   selection_time_s=study[name].get("secs"),
                   n_evaluations=len(study[name].get("history", [])),
                   best_objective=min((h["objective"]
                                       for h in study[name].get("history", [])),
                                      default=None))
        for name in OPM_DEGREES if name in study}

    m = data["_meta"]
    report = dict(
        experiment="naca_parametric",
        problem="2D NACA 0012, laminar SU2 at Re_c=1000 (angle-of-attack family)",
        parameter="angle of attack alpha (deg)",
        config=dict(
            state_dim=int(m["d"]), n_nodes=int(m["n_nodes"]), K=int(m["K"]),
            proj_residual=m["proj_residual"],
            n_train=int(data["train"]["S"].shape[1]),
            n_params_train=len(prob.TRAIN_ANGLES),
            T_dev=m["T_dev"],
            train_range=[prob.ALPHA_MIN, prob.ALPHA_MAX],
            train_angles=prob.TRAIN_ANGLES,
            val_interp_angles=prob.VAL_INTERP_ANGLES,
            val_extrap_angle=prob.VAL_EXTRAP_ANGLE,
            test_interp_angle=prob.TEST_INTERP_ANGLE,
            test_extrap_angle=prob.TEST_EXTRAP_ANGLE,
            delta=CFG["delta"], s_pool=CFG["s_pool"],
            n_grad_samples=CFG["n_grad_samples"], r_max=R_MAX,
            r_budget=R_BUDGET),
        selected_budgets=Ls,
        budget_selection=budget_selection,
        certificate_validity=certificate_validity,
        training_times_s=training_times_s,
        errors=errors,
        active_features=active_features)
    _save(REPORT, report)
    ok = all(v["all_valid"] for v in certificate_validity.values())
    print(f"    saved {REPORT.name}  (certificate <= budget for all r: {ok})",
          flush=True)
    return report


def main(phase: str = "all") -> None:
    data = make_dataset()
    S = {sp: jnp.asarray(data[sp]["S"]) for sp in SPLITS}
    m = data["_meta"]
    print(f"[naca] d={m['d']} (n_nodes={m['n_nodes']}) K={m['K']} "
          f"proj_resid={m['proj_residual']:.1e} n_train={S['train'].shape[1]} "
          f"({len(prob.TRAIN_ANGLES)} angles, t >= {m['T_dev']:g}) phase={phase}",
          flush=True)
    print(f"  train alpha:   {prob.TRAIN_ANGLES}", flush=True)
    print(f"  val_interp:    {prob.VAL_INTERP_ANGLES}   "
          f"val_extrap: {prob.VAL_EXTRAP_ANGLE}", flush=True)
    print(f"  test_interp:   {prob.TEST_INTERP_ANGLE}   "
          f"test_extrap: {prob.TEST_EXTRAP_ANGLE}", flush=True)

    if phase in ("all", "budget"):
        run_budget_selection(S)
    Ls = resolve_Ls()
    print(f"[naca] selected budgets: "
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
    print(f"[naca] elapsed {time.time() - t0:.1f}s")
