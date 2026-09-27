"""2D Boussinesq rising-thermal comparison experiment (driver).

Compares POD, Greedy-QM, Greedy-CM, Geelen-QM, and the budget-calibrated
Greedy-OPM (degrees {2} and {2, 3}) on the rising-thermal problem of
:mod:`problem`, over a range of Reynolds numbers and with **two** test
scenarios:

* ``test_interp`` -- a random interior Reynolds number (interpolation);
* ``test_extrap`` -- a Reynolds number well beyond the training range
  (extrapolation), where a loose (large-``L``) polynomial manifold
  over-extrapolates and the Lipschitz budget earns its keep.

The Lipschitz budget ``L`` is chosen **automatically and per variant**
(``p=2`` and ``p=3`` separately) by the grid-adaptive sweep in
:mod:`budget_select`: a geometric-mean objective over the interpolation and
extrapolation *validation* splits, refined to 0.01 precision.  Nothing is
hard-coded in ``methods.py``.

Phases (resumable via the JSON checkpoints):

* ``budget`` -- select ``L*_{p2}`` and ``L*_{p3}`` (grid-adaptive), record
  the full-range budget curve for the figures.
* ``sweep``  -- all methods for r = 1..R_MAX (each Greedy-OPM at its own L*).
* ``plots``  -- error-vs-r curves, active-feature bars, budget-study plot.
* ``field``  -- temperature and speed field panels (interp + extrap).

All fits and errors live in the lossless compressed coordinates of
:mod:`problem` (``K``-dimensional, exact full-state errors); the field
panels lift the reconstructions back to the ``256 x 448`` grid.
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
import problem as P  # noqa: E402
from problem import SPLITS, make_dataset  # noqa: E402
from budget_select import select_budget  # noqa: E402

PLOTS = _HERE / "plots"
CKPT = PLOTS / "boussinesq_results.json"
CKPT_BUDGET = PLOTS / "boussinesq_budget_study.json"
REPORT = PLOTS / "boussinesq_report.json"

R_MAX = 10
RS = list(range(1, R_MAX + 1))
R_BUDGET = 10
R_FIELD = 10
FIELD_FRAC = 0.7

PLOT_GRID = list(np.geomspace(1.05, 1.0e5, 30))
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
    from plots import (active_features_bar, budget_study,
                       certificate_vs_budget, error_vs_r)

    results = _load(CKPT)
    if results:
        error_vs_r(results, RS, METHOD_NAMES,
                   str(PLOTS / "boussinesq_error_vs_r"),
                   splits=("train", "test_interp", "test_extrap"))
        active_features_bar(results, RS, METHOD_NAMES,
                            str(PLOTS / "boussinesq_active_features"))
        certificate_vs_budget(results, RS, list(OPM_DEGREES), Ls,
                              str(PLOTS / "boussinesq_certificate"))
    study = _load(CKPT_BUDGET)
    if study:
        budget_study(study, str(PLOTS / "boussinesq_budget_study"))


# ---------------------------------------------------------------------------
# Phase: field panels (interp + extrap)
# ---------------------------------------------------------------------------


def run_field(data: dict, S: dict, Ls: dict) -> None:
    from plots import field_panels

    models = {name: build_model(name, S["train"], R_FIELD, L=Ls.get(name))
              for name in METHOD_NAMES}
    ext = [0.0, P.LX, 0.0, P.LY]
    for split, tag in (("test_interp", "interpolated"),
                       ("test_extrap", "extrapolated")):
        Sfull = S[split]
        # Show the *hardest* snapshot -- the one least captured by the linear
        # POD subspace (a method-agnostic difficulty measure) -- rather than a
        # fixed time index; see burgers2d for the rationale.  The annotated
        # relative error is the error over the *entire* test split (the same
        # quantity plotted in error-vs-r), not this one snapshot.
        pod = np.asarray(models["POD"].reconstruct(Sfull))
        truth = np.asarray(Sfull)
        per_col = (np.linalg.norm(pod - truth, axis=0)
                   / np.linalg.norm(truth, axis=0))
        col = int(np.argmax(per_col))
        # lift the truth and every reconstruction in one pass over the files
        cols = [truth[:, col]]
        rel = {}
        for name, model in models.items():
            cols.append(np.asarray(
                model.reconstruct(Sfull[:, col:col + 1]))[:, 0])
            rel[name] = float(model.reconstruction_error(Sfull, relative=True))
        full = P.lift(np.stack(cols, axis=1))                 # (d, 1 + n_methods)
        truth_col = full[:, 0]
        for kind, fn, cmap in (("theta", P.temperature, "inferno"),
                               ("speed", P.speed, "viridis")):
            recon = {name: fn(full[:, 1 + i])
                     for i, name in enumerate(models)}
            field_panels(
                fn(truth_col), recon, METHOD_NAMES,
                str(PLOTS / f"boussinesq_{kind}_{split}"),
                extent=ext, rel_errs=rel, xlabel=r"$x$", ylabel=r"$y$",
                aspect="equal", cmap=cmap, transpose=True,
            )
        print(f"    field panels ({tag}): snapshot column {col} "
              f"(t = {P.DT * P.STORE_EVERY_TEST * col:.2f})", flush=True)


# ---------------------------------------------------------------------------
# Phase: journal report (values worth reporting in a paper)
# ---------------------------------------------------------------------------


def run_report(data: dict, Ls: dict) -> dict:
    """Collect the journal-reportable quantities into a single JSON."""
    import problem as prob

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
            all_valid=all(x["valid"] for x in per_r),
        )

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
        experiment="boussinesq",
        problem="2D Boussinesq rising thermal",
        parameter="Reynolds number Re = U R / nu (Pr = 1)",
        config=dict(
            state_dim=int(data["_meta"]["d"]),
            compressed_dim=int(data["_meta"]["K"]),
            compression_residual=float(data["_meta"]["proj_residual"]),
            n_train=int(data["train"]["S"].shape[1]),
            n_params_train=len(data["train"]["Re"]),
            grid=[prob.NX, prob.NY], box=[prob.LX, prob.LY],
            t_end=prob.T_END, dt=prob.DT,
            train_range=[prob.RE_MIN, prob.RE_MAX],
            train_reynolds=data["train"]["Re"],
            val_interp_reynolds=data["val_interp"]["Re"],
            val_extrap_reynolds=data["val_extrap"]["Re"][0],
            test_interp_reynolds=data["test_interp"]["Re"][0],
            test_extrap_reynolds=data["test_extrap"]["Re"][0],
            delta=CFG["delta"], s_pool=CFG["s_pool"],
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
    print(f"[boussinesq] d={data['_meta']['d']} (compressed K="
          f"{data['_meta']['K']}, residual {data['_meta']['proj_residual']:.1e}) "
          f"n_train={S['train'].shape[1]} "
          f"({len(data['train']['Re'])} Re x "
          f"{S['train'].shape[1] // len(data['train']['Re'])}) "
          f"delta={CFG['delta']:g} phase={phase}", flush=True)
    print(f"  train Re:    {np.round(data['train']['Re'], 1).tolist()}",
          flush=True)
    print(f"  val_interp:  {np.round(data['val_interp']['Re'], 1).tolist()}"
          f"   val_extrap: {data['val_extrap']['Re'][0]:.1f}", flush=True)
    print(f"  test_interp: {data['test_interp']['Re'][0]:.1f}"
          f"   test_extrap: {data['test_extrap']['Re'][0]:.1f}", flush=True)

    if phase in ("all", "budget"):
        run_budget_selection(S)
    Ls = resolve_Ls()
    print(f"[boussinesq] selected budgets: "
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
    print(f"[boussinesq] elapsed {time.time() - t0:.1f}s")
