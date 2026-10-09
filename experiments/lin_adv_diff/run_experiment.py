"""Advection-diffusion comparison experiment (driver).

Compares POD, Greedy-QM, Geelen-QM, and the budget-calibrated Greedy-OPM
(degrees {2} and {2, 3}) on the periodic linear advection-diffusion problem
of :mod:`problem`, parametric in the **initial pulse width** and with **two**
test scenarios:

* ``test_interp`` -- a random interior pulse width (interpolation);
* ``test_extrap`` -- a pulse narrower than any training pulse (extrapolation),
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
* ``cpn``    -- CPN-S (released code) tolerance sweep, one run per n.
* ``noise`` -- robustness of the r = R_NOISE models to additive white noise
  on the states (error vs noise-to-signal ratio).
* ``plots``  -- error-vs-r curves, active-feature bars, budget-study plot.
* ``field``  -- space-time reconstruction / error panels (interp + extrap).
* ``all``    -- everything above, in order.
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
import jax.numpy as jnp  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from methods import (  # noqa: E402
    BAR_METHODS,
    CFG,
    CPN_CFG,
    CPN_NAME,
    METHOD_NAMES,
    OPM_DEGREES,
    PLOT_METHODS,
    active_by_degree,
    build_cpn,
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

#: CPN-S tolerance sweep: geometric grid from CPN_TOL_HI downwards
#: (CPN_PER_DECADE points per decade), stopped once the adaptive rule
#: promotes more than R_MAX modes, then bisected to hit every n <= R_MAX.
CPN_TOL_HI = 0.9

#: Noise robustness study: the models fitted at R_NOISE (training dilation
#: CFG["delta"]) reconstruct the training states corrupted by additive white
#: noise at noise-to-signal ratio sigma (entrywise, relative to the RMS of the
#: centered training snapshots).
NOISE_SIGMAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
R_NOISE = R_FIELD
NOISE_SEED = 0
CPN_TOL_LO = 1.0e-6
CPN_PER_DECADE = 4

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
    """Atomic JSON checkpoint (write to a temporary file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=0))
    os.replace(tmp, path)


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
# Phase: CPN-S tolerance sweep (released code, same POD pool)
# ---------------------------------------------------------------------------


def run_cpn(S: dict) -> dict:
    """Sweep the CPN-S target tolerance; keep one manifold per achieved n.

    CPN-S (Bensalah, Nouy & Soffo 2026) chooses its own dimension ``n`` from
    the target tolerance, so it cannot be fitted at a prescribed ``r``.  The
    sweep walks the tolerance down from ``CPN_TOL_HI`` until more than
    ``R_MAX`` modes get promoted, bisects the gaps so that every ``n <= R_MAX``
    reachable by the adaptive rule is hit, and stores, per ``n``, the run with
    the tightest tolerance as ``CPN-S@r=n``.  Every run of the sweep is kept
    under ``_cpn_sweep`` in the checkpoint.
    """
    results = _load(CKPT)
    if results.get("_cpn_sweep", {}).get("complete"):
        print("  [cpn] cached tolerance sweep "
              f"({len(results['_cpn_sweep']['runs'])} runs)", flush=True)
        return results
    from cpn_baseline import CPN_COMMIT, sweep_tolerances

    def build(tol):
        return build_cpn(S["train"], tol, n_max=R_MAX)

    def evaluate(pm):
        return _evaluate(pm, S)

    prior = results.get("_cpn_sweep", {}).get("runs", [])
    elapsed = float(results.get("_cpn_sweep", {}).get("secs", 0.0))

    def checkpoint(runs):
        results["_cpn_sweep"] = dict(complete=False, runs=runs,
                                     secs=round(elapsed + time.time() - tic, 1))
        _save(CKPT, results)

    tic = time.time()
    sw = sweep_tolerances(build, evaluate, n_max=R_MAX, tol_hi=CPN_TOL_HI,
                          tol_lo=CPN_TOL_LO, per_decade=CPN_PER_DECADE,
                          log=print, prior_runs=prior, checkpoint=checkpoint)
    for n, rec in sw["by_n"].items():
        if n not in RS:
            continue
        results[f"{CPN_NAME}@r={n}"] = dict(
            method=CPN_NAME, r=int(n), **{sp: rec[sp] for sp in SPLITS},
            n_active=int(rec["n_active_nonlinear"]),
            n_active_by_degree=rec["n_active_by_degree"],
            certificate=None, lambda_=None, secs=rec["fit_secs"],
            tol=rec["tol"], n_nonzero_coeffs=rec["n_nonzero_coeffs"],
            n_learnt_tail=rec["n_learnt_tail"],
            cpn_lipschitz=rec["cpn_lipschitz"],
            pod_floor_rel=rec["pod_floor_rel"])
    results["_cpn_sweep"] = dict(
        complete=True, config=dict(CPN_CFG, n_max=R_MAX, tol_hi=CPN_TOL_HI,
                                   tol_lo=CPN_TOL_LO, per_decade=CPN_PER_DECADE,
                                   cpn_commit=CPN_COMMIT),
        runs=sw["runs"], secs=round(elapsed + time.time() - tic, 1))
    _save(CKPT, results)
    got = sorted(int(n) for n in sw["by_n"] if int(n) in RS)
    print(f"  [cpn] achieved n = {got}  ({results['_cpn_sweep']['secs']}s)",
          flush=True)
    return results


# ---------------------------------------------------------------------------
# Phase: plots
# ---------------------------------------------------------------------------


def run_plots(Ls: dict) -> None:
    """Render every figure (journal-quality, no titles) from the checkpoints."""
    from plots import (active_features_bar, budget_study,
                       certificate_vs_budget, error_vs_r)

    results = _load(CKPT)
    error_vs_r(results, RS, PLOT_METHODS,
               str(PLOTS / "lin_adv_diff_error_vs_r"),
               splits=("test_interp", "test_extrap"))
    active_features_bar(results, RS, BAR_METHODS,
                        str(PLOTS / "lin_adv_diff_active_features"))
    certificate_vs_budget(results, RS, list(OPM_DEGREES), Ls,
                          str(PLOTS / "lin_adv_diff_certificate"))
    if results.get("_noise", {}).get("complete"):
        from plots import noise_study
        noise_study(results["_noise"], PLOT_METHODS,
                    str(PLOTS / "lin_adv_diff_noise"))
    study = _load(CKPT_BUDGET)
    if study:
        budget_study(study, str(PLOTS / "lin_adv_diff_budget_study"))


# ---------------------------------------------------------------------------
# Phase: field panels (interp + extrap)
# ---------------------------------------------------------------------------


def _field_models(S: dict, Ls: dict) -> dict:
    """Fitted models at ``R_FIELD`` in plot order, CPN-S included when the
    tolerance sweep reached ``n = R_FIELD`` (refit at that tolerance)."""
    results = _load(CKPT)
    models = {}
    for name in PLOT_METHODS:
        if name == CPN_NAME:
            rec = results.get(f"{CPN_NAME}@r={R_FIELD}")
            if rec is None:
                print(f"  [field] no CPN-S run with n = {R_FIELD}; skipped",
                      flush=True)
                continue
            pm = build_cpn(S["train"], rec["tol"], n_max=R_FIELD)
            assert not pm.capped_ and pm.n_ == R_FIELD, (pm.capped_, pm.n_)
            models[name] = pm
        else:
            models[name] = build_model(name, S["train"], R_FIELD,
                                       L=Ls.get(name))
    return models


def _field_cache(split: str) -> Path:
    return PLOTS / f"lin_adv_diff_field_{split}.npz"


def _replot_field(field_kw: dict) -> None:
    """Re-render the field panels from the cached images (no refit)."""
    from plots import field_panels

    for split in ("test_interp", "test_extrap"):
        z = np.load(_field_cache(split), allow_pickle=True)
        names = list(z["names"])
        rel = {nm: float(v) for nm, v in zip(names, z["rel"])}
        recon = {nm: z[f"recon_{i}"] for i, nm in enumerate(names)}
        field_panels(z["truth"], recon, names,
                     str(PLOTS / f"lin_adv_diff_field_{split}"),
                     extent=list(z["extent"]), rel_errs=rel, **field_kw)
        print(f"    field panels ({split}) re-plotted from the cache", flush=True)


def run_field(data: dict, S: dict, Ls: dict, replot: bool = False) -> None:
    """Space-time reconstruction / error panels on both test splits.  The
    images are cached in ``plots/lin_adv_diff_field_<split>.npz`` so the
    panels can be re-plotted without a refit (``run_experiment.py fieldplot``).
    """
    from plots import field_panels

    if replot:
        _replot_field({})
        return
    models = _field_models(S, Ls)
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
            truth, recon, list(models),
            str(PLOTS / f"lin_adv_diff_field_{split}"),
            extent=extent, rel_errs=rel,
        )
        np.savez_compressed(
            _field_cache(split), names=np.array(list(models)),
            rel=np.array([rel[nm] for nm in models]), extent=np.array(extent),
            truth=truth, **{f"recon_{i}": recon[nm] for i, nm in enumerate(models)})


# ---------------------------------------------------------------------------
# Phase: noise robustness (additive white noise on the training states)
# ---------------------------------------------------------------------------


def run_noise(S: dict, Ls: dict, d_full: int) -> dict:
    """Reconstruction error under additive white noise on the states.

    Every model (fitted once at ``r = R_NOISE`` with the training dilation
    ``CFG["delta"]``, CPN-S at its ``n = R_NOISE`` tolerance) encodes and
    decodes the noisy training states

        S_sigma = S_train + sigma * rms * Z,   Z_ij ~ N(0, 1) i.i.d.,

    where ``rms = ||S_train - mu||_F / sqrt(d_full n)`` is the entrywise RMS
    of the centered training snapshots in the full state space (so ``sigma``
    is the noise-to-signal ratio per entry) and ``Z`` is one fixed draw
    (``NOISE_SEED``) shared by all methods and all ``sigma``.  Two errors are
    recorded: against the clean states, ``||S_train - G(E(S_sigma))|| /
    ||S_train - mu||`` (how much of the noise the manifold lets through),
    and against the noisy inputs, ``||S_sigma - G(E(S_sigma))|| /
    ||S_sigma - mu||`` (a plain encode-decode of the noisy data, which
    carries the irreducible off-subspace noise for every method).  When the
    states are compressed coordinates (Boussinesq) the noise is drawn in the
    compressed space with the same entrywise standard deviation, which is
    what full-state white noise looks like after the orthonormal
    compression; its component orthogonal to the compressed space is
    identical for all methods and only enters the second error.
    """
    results = _load(CKPT)
    rec = results.get("_noise", {})
    if (rec.get("complete") and rec.get("r") == R_NOISE
            and rec.get("sigmas") == NOISE_SIGMAS
            and rec.get("seed") == NOISE_SEED):
        print("  [noise] cached", flush=True)
        return results
    models = _field_models(S, Ls)
    S_tr = np.asarray(S["train"])
    mu = S_tr.mean(axis=1, keepdims=True)
    n = S_tr.shape[1]
    rms = float(np.linalg.norm(S_tr - mu) / np.sqrt(float(d_full) * n))
    Z = np.random.default_rng(NOISE_SEED).standard_normal(S_tr.shape)
    err_clean = {name: [] for name in models}
    err_noisy = {name: [] for name in models}
    denom_clean = float(np.linalg.norm(S_tr - mu))
    # Noise energy outside the stored state space (nonzero only when the
    # states are compressed coordinates): full-state white noise has an
    # expected energy sigma^2 rms^2 (d_full - d_state) n orthogonal to the
    # compressed space, which no method sees or can represent; it is added
    # analytically to the error against the noisy inputs so that this error
    # means the same thing for compressed and uncompressed examples.
    d_state = int(S_tr.shape[0])
    for sigma in NOISE_SIGMAS:
        S_sig = S_tr + float(sigma) * rms * Z
        extra = (float(sigma) * rms) ** 2 * (int(d_full) - d_state) * n
        denom_noisy = float(np.sqrt(np.linalg.norm(S_sig - mu) ** 2 + extra))
        for name, pm in models.items():
            R = np.asarray(pm.reconstruct(S_sig))
            err_clean[name].append(float(np.linalg.norm(S_tr - R) / denom_clean))
            err_noisy[name].append(float(
                np.sqrt(np.linalg.norm(S_sig - R) ** 2 + extra) / denom_noisy))
        print(f"  [noise] sigma={sigma:g}  "
              + "  ".join(f"{name}={err_clean[name][-1]:.2e}" for name in models),
              flush=True)
    results["_noise"] = dict(
        complete=True, r=R_NOISE, sigmas=NOISE_SIGMAS, seed=NOISE_SEED,
        d_full=int(d_full), d_state=d_state, rms=rms, delta_train=CFG["delta"],
        err_clean=err_clean, err_noisy=err_noisy,
        definition="S_sigma = S_train + sigma * rms * Z with Z ~ N(0,1) i.i.d. "
                   "(fixed seed), rms = ||S_train - mu||_F / sqrt(d_full n); "
                   "err_clean = ||S_train - G(E(S_sigma))|| / ||S_train - mu||, "
                   "err_noisy = ||S_sigma - G(E(S_sigma))|| / ||S_sigma - mu|| with "
                   "the expected off-state-space noise energy added to both norms",
    )
    results.pop("_perturbation", None)
    results.pop("_dilation", None)
    _save(CKPT, results)
    return results


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

    # CPN-S: tolerance sweep and the per-r selections
    cpn = results.get("_cpn_sweep", {})
    cpn_keys = ("tol", "n", "capped", "fit_secs", "train", "val_interp",
                "val_extrap", "test_interp", "test_extrap",
                "n_active_nonlinear", "n_active_by_degree", "n_nonzero_coeffs",
                "n_learnt_tail", "cpn_lipschitz", "pod_floor_rel")
    cpn_section = dict(
        settings=cpn.get("config"),
        sweep_time_s=cpn.get("secs"),
        sweep=[{k: run.get(k) for k in cpn_keys if k in run}
               for run in cpn.get("runs", [])],
        selected_by_r={
            str(r): {k: results[f"{CPN_NAME}@r={r}"].get(k)
                     for k in ("tol", *SPLITS, "n_active",
                               "n_active_by_degree", "n_nonzero_coeffs",
                               "n_learnt_tail", "cpn_lipschitz", "secs")}
            for r in RS if f"{CPN_NAME}@r={r}" in results},
    )
    for r in RS:
        rec = results.get(f"{CPN_NAME}@r={r}")
        if rec is not None:
            errors.setdefault(CPN_NAME, {})[str(r)] = {sp: rec[sp] for sp in SPLITS}
            active_features.setdefault(CPN_NAME, {})[str(r)] = rec["n_active"]
            training_times_s.setdefault(CPN_NAME, {})[str(r)] = rec["secs"]

    report = dict(
        experiment="lin_adv_diff",
        problem="periodic linear advection-diffusion",
        parameter="initial pulse width w (squared Gaussian width)",
        config=dict(
            state_dim=int(data["train"]["S"].shape[0]),
            n_train=int(data["train"]["S"].shape[1]),
            n_params_train=len(data["train"]["widths"]),
            snaps_per_param=data["train"]["n_per"],
            train_range=[prob.W_MIN, prob.W_MAX],
            train_widths=data["train"]["widths"],
            val_interp_widths=data["val_interp"]["widths"],
            val_extrap_width=data["val_extrap"]["w"],
            test_interp_width=data["test_interp"]["w"],
            test_extrap_width=data["test_extrap"]["w"],
            wave_speed=prob.C,
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
        cpn=cpn_section,
        noise=results.get("_noise"),
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
          f"({len(data['train']['widths'])} widths x {data['train']['n_per']}) "
          f"delta={CFG['delta']:g} phase={phase}", flush=True)
    print(f"  train widths:  {[f'{w:.2e}' for w in data['train']['widths']]}",
          flush=True)
    print(f"  val_interp:    {[f'{w:.2e}' for w in data['val_interp']['widths']]}"
          f"   val_extrap: {data['val_extrap']['w']:.2e}", flush=True)
    print(f"  test_interp:   {data['test_interp']['w']:.2e}"
          f"   test_extrap: {data['test_extrap']['w']:.2e}", flush=True)

    if phase in ("all", "budget"):
        run_budget_selection(S)
    Ls = resolve_Ls()
    print(f"[lin_adv_diff] selected budgets: "
          + ", ".join(f"{k} -> L*={v:g}" for k, v in Ls.items()), flush=True)
    if phase in ("all", "sweep"):
        run_sweep(S, Ls)
    if phase in ("all", "cpn"):
        run_cpn(S)
    if phase in ("all", "noise"):
        run_noise(S, Ls, int(S["train"].shape[0]))
    if phase in ("all", "plots"):
        run_plots(Ls)
    if phase in ("all", "field"):
        run_field(data, S, Ls)
    if phase == "fieldplot":
        run_field(data, S, Ls, replot=True)
    if phase in ("all", "plots", "report"):
        run_report(data, Ls)
    print("DONE")


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[lin_adv_diff] elapsed {time.time() - t0:.1f}s")
