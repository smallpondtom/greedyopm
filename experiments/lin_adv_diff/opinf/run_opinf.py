"""Driver: standard vs geometric manifold-Galerkin OpInf on learned manifolds.

Pipeline (see ``notes/opinf_opm.tex`` and the README):

1.  Fit the six manifolds -- POD, Greedy-QM, Greedy-CM, Geelen-QM, and
    Greedy-OPM (p=2)/(p=3) -- on the training snapshot matrix of the
    reconstruction experiment, with the **same settings** (configuration and
    the reconstruction-selected Lipschitz budgets ``L*`` read from
    ``../plots/lin_adv_diff_budget_study.json``).
2.  For every manifold, learn one reduced operator set per training wave speed,
    twice: (1) **standard** latent OpInf and (2) **GMG** OpInf.  The standard
    model is evaluated even where it diverges -- demonstrating that it does is
    part of the experiment.
3.  Simulate every ROM (RK4 on the dense trajectory grid), decode through the
    manifold, and record the relative Frobenius state error on the training
    speeds and on the held-out **interpolated** speed (spline-interpolated
    operators).  Extrapolated speeds are not evaluated.

Phases (resumable JSON checkpoints): ``sweep``, ``plots``, ``report``, ``all``.
Usage:  python run_opinf.py [phase]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))                    # problem, methods, ...
sys.path.insert(0, str(_HERE.parent.parent.parent / "src"))  # greedyopm

from greedyopm import opinf                              # noqa: E402
from opinf_methods import OPINF_METHODS, OPM_NAMES, ManifoldROM  # noqa: E402
import problem_opinf as P                                # noqa: E402
from budget_select_rom import (select_budget_rom,        # noqa: E402
                               select_budget_rom_per_r)

PLOTS = _HERE / "plots"
#: Problem configuration ("gentle" default; "manifold" = the reconstruction
#: experiment's harsh settings).  Select with the OPINF_REGIME env var.
REGIME = os.environ.get("OPINF_REGIME", "gentle")
#: Operator-learning variant (OPINF_VARIANT env var):
#:   "affine" (default) -- pooled affine-in-parameter operators
#:       ``O(c) = O0 + tau O1`` fit on ALL training speeds jointly, with
#:       EXACT spectral velocities as regression targets (the FOM operator is
#:       affine in the speed, so this parametric form is exact);
#:   "legacy" -- one operator set per training speed from FBCT4
#:       finite-difference velocities, entrywise natural-cubic-spline
#:       interpolation at a query speed (the original pipeline; the archived
#:       "manifold"-regime results used this variant).
VARIANT = os.environ.get("OPINF_VARIANT", "affine")
if VARIANT not in ("affine", "legacy"):
    raise ValueError(f"unknown OPINF_VARIANT {VARIANT!r}")
TAG = f"opinf_{REGIME}" + ("_affine" if VARIANT == "affine" else "")
CKPT = PLOTS / f"{TAG}_results.json"
CKPT_B = PLOTS / f"{TAG}_rom_budget.json"
REPORT = PLOTS / f"{TAG}_report.json"
R_BUDGET = 10
BUDGET_JSON = _HERE.parent / "plots" / "lin_adv_diff_budget_study.json"

R_MAX = 10
RS = list(range(1, R_MAX + 1))
KINDS = ("plain", "gmg")
SPLINE_ORDER = 3          # natural cubic spline of the per-speed operators
DIVERGE_TOL = 10.0        # a relative error above this counts as diverged
STORE_CAP = 1.0e30        # JSON-safe cap for diverged error magnitudes


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=0))


def load_recon_budgets() -> dict:
    """The manifold experiment's reconstruction-selected budgets.

    Only meaningful for the ``manifold`` regime (the gentle regime has no
    corresponding reconstruction experiment); empty otherwise.
    """
    if REGIME != "manifold" or not BUDGET_JSON.exists():
        return {}
    study = json.loads(BUDGET_JSON.read_text())
    return {name: float(study[name]["L_star"]) for name in OPM_NAMES}


def run_budget(data: dict) -> dict:
    """Select each OPM variant's budget against the reduced-model error."""
    study = _load(CKPT_B)
    for name in OPM_NAMES:
        if name in study and "L_star" in study[name]:
            print(f"  [budget] {name}: cached ROM L* = "
                  f"{study[name]['L_star']:g}", flush=True)
            continue
        tic = time.time()
        rec = select_budget_rom(name, R_BUDGET, data, kind="gmg", log=print)
        rec["secs"] = round(time.time() - tic, 1)
        study[name] = rec
        _save(CKPT_B, study)
    # per-r refinement around the r=10 anchor (ROM stability is per-r)
    for name in OPM_NAMES:
        if "per_r" in study[name]:
            print(f"  [budget] {name}: cached per-r budgets", flush=True)
            continue
        tic = time.time()

        def _cb(partial, _name=name):
            study[_name].update(partial)
            _save(CKPT_B, study)

        per = select_budget_rom_per_r(
            name, RS, data, anchor_L=study[name]["L_star"], kind="gmg",
            log=print, done=study[name].get("per_r_partial"),
            extra_refs=[study[name]["L_refit"]]
            if "L_refit" in study[name] else (),
            checkpoint_cb=lambda partial, _n=name: (
                study[_n].__setitem__("per_r_partial", partial["per_r"]),
                _save(CKPT_B, study)))
        per["per_r_secs"] = round(time.time() - tic, 1)
        study[name].pop("per_r_partial", None)
        study[name].update(per)
        _save(CKPT_B, study)
    return study


def resolve_rom_budgets() -> dict:
    """Per-r ROM budgets ``{method: {r: L*(r)}}`` for the OPM variants."""
    study = _load(CKPT_B)
    Ls = {}
    for name in OPM_NAMES:
        if name not in study or "per_r" not in study[name]:
            raise RuntimeError(
                "ROM budgets not selected; run the 'budget' phase.")
        Ls[name] = {int(k): dict(v)
                    for k, v in study[name]["per_r"].items()}
    return Ls


def _budget_for(Ls: dict, name: str, r: int):
    entry = Ls.get(name)
    if entry is None:
        return None
    return entry[int(r)] if isinstance(entry, dict) else float(entry)


def _finite(err: float) -> float:
    return float(min(err, STORE_CAP)) if np.isfinite(err) else STORE_CAP


def _interp_ops(speeds, op_list, c_query, kind):
    """Spline-interpolate a per-speed operator list at an interior speed."""
    if kind == "gmg":
        out = {"a": opinf.interp_operator_dict(
            speeds, [d["a"] for d in op_list], c_query, order=SPLINE_ORDER)}
        if "b" in op_list[0]:
            out["b"] = opinf.interp_operator_dict(
                speeds, [d["b"] for d in op_list], c_query, order=SPLINE_ORDER)
        return out
    return opinf.interp_operator_dict(speeds, op_list, c_query,
                                      order=SPLINE_ORDER)


def _learn_all(mrom: ManifoldROM, data: dict, kind: str):
    """Learn the parametric operator family for one manifold + ROM kind."""
    if VARIANT == "affine":
        return mrom.learn_affine(data["train"]["blocks"],
                                 data["train"]["dblocks"],
                                 data["train"]["speeds"], kind)
    return [mrom.learn(blk, data["dt"], kind)
            for blk in data["train"]["blocks"]]


def _ops_for(mrom: ManifoldROM, learned, data: dict, c: float, kind: str):
    """Operators at speed ``c`` from the learned parametric family."""
    if VARIANT == "affine":
        return mrom.ops_at(learned, c, kind)
    return _interp_ops(data["train"]["speeds"], learned, c, kind)


def _build_mrom(name: str, r: int, data: dict, L) -> ManifoldROM:
    """Construct the manifold exactly as the budget selection evaluated it:
    a plain budget fits directly; a ``{L_ref, L}`` pair fits at the reference
    budget and recalibrates via ``set_budget`` (mode selection preserved)."""
    if isinstance(L, dict):
        mrom = ManifoldROM(name, r, data["train"]["S_fit"], L=L["L_ref"])
        return mrom.set_budget(L["L"])
    return ManifoldROM(name, r, data["train"]["S_fit"], L=L)


def evaluate_method(name: str, r: int, data: dict, L) -> dict:
    """Fit the manifold and score both ROM kinds on train + interp speeds."""
    tic = time.time()
    mrom = _build_mrom(name, r, data, L)
    fit_secs = round(time.time() - tic, 1)

    dt = data["dt"]
    speeds = data["train"]["speeds"]
    blocks = data["train"]["blocks"]
    S_int = data["interp"]["S"]

    rec = dict(method=name, r=r, fit_secs=fit_secs,
               n_features=int(mrom.n_features),
               certificate=(float(mrom.certificate)
                            if np.isfinite(mrom.certificate) else None),
               recon_interp=_finite(mrom.manifold_reconstruction_error(S_int)))

    for kind in KINDS:
        jax.clear_caches()
        tic = time.time()
        learned = _learn_all(mrom, data, kind)
        train_errs = [
            opinf.relative_state_error(
                blk, mrom.simulate(_ops_for(mrom, learned, data, c_t, kind),
                                   blk, dt, kind))
            for c_t, blk in zip(speeds, blocks)
        ]
        ops_i = _ops_for(mrom, learned, data, data["interp"]["c"], kind)
        err_i = opinf.relative_state_error(
            S_int, mrom.simulate(ops_i, S_int, dt, kind))
        rec[kind] = dict(
            train_per_speed=[_finite(e) for e in train_errs],
            train_median=_finite(float(np.median(train_errs))),
            interp=_finite(err_i),
            diverged_train=bool(any((not np.isfinite(e)) or e > DIVERGE_TOL
                                    for e in train_errs)),
            diverged_interp=bool((not np.isfinite(err_i))
                                 or err_i > DIVERGE_TOL),
            secs=round(time.time() - tic, 1),
        )
    return rec


def run_sweep(data: dict, Ls: dict) -> dict:
    results = _load(CKPT)
    results.setdefault("_meta", {}).update(
        speeds=data["train"]["speeds"], c_interp=data["interp"]["c"],
        n_time=data["n_time"], dt=data["dt"], L_star_rom=Ls,
        L_star_recon=load_recon_budgets(), variant=VARIANT,
        diverge_tol=DIVERGE_TOL, spline_order=SPLINE_ORDER)
    _save(CKPT, results)
    for r in RS:
        for name in OPINF_METHODS:
            key = f"{name}@r={r}"
            if key in results:
                continue
            rec = evaluate_method(name, r, data, _budget_for(Ls, name, r))
            results[key] = rec
            _save(CKPT, results)
            pl, gm = rec["plain"], rec["gmg"]
            print(f"  r={r:2d} {name:18s} "
                  f"plain: tr={pl['train_median']:.2e} in={pl['interp']:.2e} | "
                  f"gmg: tr={gm['train_median']:.2e} in={gm['interp']:.2e} "
                  f"(fit {rec['fit_secs']}s, rom {pl['secs']}+{gm['secs']}s)",
                  flush=True)
    return results


def run_plots(data: dict, Ls: dict) -> None:
    import plots_opinf as PL
    results = _load(CKPT)
    PL.error_vs_r(results, RS, OPINF_METHODS, "plain",
                  str(PLOTS / f"{TAG}_error_vs_r_plain"))
    PL.error_vs_r(results, RS, OPINF_METHODS, "gmg",
                  str(PLOTS / f"{TAG}_error_vs_r_gmg"))
    PL.fields(data, results, OPINF_METHODS, Ls,
              str(PLOTS / f"{TAG}_field"))


def run_report(data: dict, Ls: dict) -> dict:
    results = _load(CKPT)
    errors = {}
    for name in OPINF_METHODS:
        errors[name] = {}
        for r in RS:
            rec = results.get(f"{name}@r={r}")
            if rec is None:
                continue
            errors[name][str(r)] = {
                kind: dict(train_median=rec[kind]["train_median"],
                           train_per_speed=rec[kind]["train_per_speed"],
                           interp=rec[kind]["interp"],
                           diverged_train=rec[kind]["diverged_train"],
                           diverged_interp=rec[kind]["diverged_interp"])
                for kind in KINDS
            }
            errors[name][str(r)]["recon_interp"] = rec["recon_interp"]
            errors[name][str(r)]["n_features"] = rec["n_features"]
    budget_study = _load(CKPT_B)
    rep = dict(
        experiment="lin_adv_diff_opinf", regime=REGIME,
        rom_budget_selection={
            n: dict(L_star=float(budget_study[n]["L_star"]),
                    best_objective=float(budget_study[n]["best_objective"]),
                    secs=budget_study[n].get("secs"))
            for n in OPM_NAMES if n in budget_study},
        problem="periodic linear advection-diffusion (manifold-problem settings)",
        rom_kinds=dict(
            plain="standard latent OpInf: shat' = c + A shat + N p(shat)",
            gmg=("geometric manifold-Galerkin OpInf: "
                 "shat' = M(shat)^{-1} (a + J_p^T b), M = I + J_p^T Vbar^T Vbar J_p")),
        config=dict(
            d=int(data["train"]["blocks"][0].shape[0]),
            train_speeds=data["train"]["speeds"],
            c_interp=data["interp"]["c"],
            n_time=data["n_time"], dt=data["dt"], T=data["t_window"],
            fit_cols=int(data["train"]["S_fit"].shape[1]),
            r_max=R_MAX, spline_order=SPLINE_ORDER,
            diverge_tol=DIVERGE_TOL,
            budgets_rom=Ls, budgets_recon=load_recon_budgets()),
        errors=errors)
    _save(REPORT, rep)
    print(f"    saved {REPORT.name}", flush=True)
    return rep


def main(phase: str = "all") -> None:
    data = P.make_opinf_dataset(REGIME)
    print(f"[opinf] {P.summary(data)}", flush=True)
    from opinf_methods import ROM_DELTA
    print(f"[opinf] ROM certificate collar delta = {ROM_DELTA:g}", flush=True)
    rb = load_recon_budgets()
    if rb:
        print(f"[opinf] reconstruction-selected budgets: "
              + ", ".join(f"{k} -> {v:g}" for k, v in rb.items()), flush=True)
    if phase in ("all", "budget"):
        run_budget(data)
    Ls = resolve_rom_budgets()
    print(f"[opinf] ROM-selected per-r budgets (note Sec. 6): "
          + ", ".join(f"{k}: r10 -> {v[10]['L']:g}" for k, v in Ls.items()),
          flush=True)
    if phase in ("all", "sweep"):
        run_sweep(data, Ls)
    if phase in ("all", "plots"):
        run_plots(data, Ls)
    if phase in ("all", "plots", "report"):
        run_report(data, Ls)
    print("DONE", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[opinf] elapsed {time.time() - t0:.1f}s")
