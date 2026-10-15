"""Field panels that are consistent with the error-vs-n curves.

The field panels show the reconstruction of *one* fitted model per method at
``n = R_FIELD``.  Refitting a model for the panels is not guaranteed to
reproduce the model of the r-sweep: Plain Greedy (p=3) with the full
degree-{2,3} monomial dictionary is so ill-conditioned (condition number
above 1e12 at n = 10) that two fits on the same machine differ by orders of
magnitude in their test error.  Therefore the field phase

1. fits each method once at ``R_FIELD`` and **writes that fit's errors back
   into the sweep entry** ``<name>@r=R_FIELD`` of the results checkpoint, so
   the error-vs-n point, the timing row and the panel come from one model;
2. caches the raw reconstructions it needs (``<exp>_field_models.npz``), so
   the panels can be redrawn, and other degree folders (p4, p5) can reuse the
   same reconstructions for every method other than CPN-S
   (``run_experiment.py field --field-from p3``).

CPN-S is deterministic given its tolerance and is refit per folder.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

SPLITS_FIELD = ("test_interp", "test_extrap")


# ---------------------------------------------------------------------------
# cache helpers
# ---------------------------------------------------------------------------


def load_field_cache(path: Path) -> dict:
    """``{split: {"col": int|None, "truth": arr, "recon": {name: arr}}}``."""
    cache = {sp: dict(col=None, truth=None, recon={}) for sp in SPLITS_FIELD}
    if not path.exists():
        return cache
    z = np.load(path, allow_pickle=True)
    names = json.loads(str(z["names_json"]))
    for sp in SPLITS_FIELD:
        if f"col__{sp}" in z:
            cache[sp]["col"] = int(z[f"col__{sp}"])
        if f"truth__{sp}" in z:
            cache[sp]["truth"] = z[f"truth__{sp}"]
        for i, nm in enumerate(names):
            key = f"recon__{sp}__{i}"
            if key in z:
                cache[sp]["recon"][nm] = z[key]
    return cache


def save_field_cache(path: Path, cache: dict) -> None:
    names = sorted({nm for sp in SPLITS_FIELD for nm in cache[sp]["recon"]})
    out = dict(names_json=np.array(json.dumps(names)))
    for sp in SPLITS_FIELD:
        if cache[sp]["col"] is not None:
            out[f"col__{sp}"] = np.array(int(cache[sp]["col"]))
        if cache[sp]["truth"] is not None:
            out[f"truth__{sp}"] = np.asarray(cache[sp]["truth"])
        for i, nm in enumerate(names):
            if nm in cache[sp]["recon"]:
                out[f"recon__{sp}__{i}"] = np.asarray(cache[sp]["recon"][nm])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(path)


def copy_from_tag(src_cache_path: Path, src_results_path: Path, cache: dict,
                  results: dict, *, R: int, method_names, cpn_name: str,
                  log=print) -> None:
    """Reuse another folder's fitted models (every method except CPN-S):
    copy its ``<name>@r=R`` sweep entries and its cached reconstructions."""
    src_res = json.loads(src_results_path.read_text())
    src_cache = load_field_cache(src_cache_path)
    n_copied = 0
    for name in method_names:
        key = f"{name}@r={R}"
        if key in src_res:
            results[key] = src_res[key]
        for sp in SPLITS_FIELD:
            if name in src_cache[sp]["recon"]:
                cache[sp]["recon"][name] = src_cache[sp]["recon"][name]
                n_copied += 1
    for sp in SPLITS_FIELD:
        if src_cache[sp]["col"] is not None:
            cache[sp]["col"] = src_cache[sp]["col"]
        if src_cache[sp]["truth"] is not None:
            cache[sp]["truth"] = src_cache[sp]["truth"]
    cache_meta = {k: v for k, v in src_res.get("_meta", {}).items()
                  if k == "field_cols"}
    results.setdefault("_meta", {}).update(cache_meta)
    log(f"  [field] reused {n_copied} cached reconstructions and the "
        f"r={R} sweep entries of {src_results_path.parent.name} "
        f"(all methods but {cpn_name})")


# ---------------------------------------------------------------------------
# the fits
# ---------------------------------------------------------------------------


def consistent_models(S: dict, Ls: dict, results: dict, cached: set, *,
                      R: int, plot_methods, cpn_name: str, build_model,
                      build_cpn, evaluate, n_active, active_by_degree, sync,
                      log=print) -> dict:
    """Fit, at ``R``, every plotted method that has no cached reconstruction.

    For the registry methods the fit's errors, timings and feature counts
    REPLACE the sweep entry ``<name>@r=R`` in ``results`` (the caller saves
    the checkpoint), so that the error-vs-n curve and the panel show the same
    model.  CPN-S is refit at its stored tolerance and its entry is left as
    it is (the refit is deterministic).  Returns ``{name: fitted model}``.
    """
    models = {}
    for name in plot_methods:
        if name in cached:
            continue
        if name == cpn_name:
            rec = results.get(f"{cpn_name}@r={R}")
            if rec is None:
                log(f"  [field] no CPN-S run with n = {R}; skipped")
                continue
            pm = build_cpn(S["train"], rec["tol"], n_max=R)
            if pm.capped_ or pm.n_ != R:
                log(f"  [field] WARNING: CPN-S refit at tol={rec['tol']:.3e} "
                    f"returned n={pm.n_} (capped={pm.capped_}), not {R}; "
                    f"rerun the cpn phase. CPN-S left out of the panels.")
                continue
            models[name] = pm
            continue
        tic = time.time()
        model = build_model(name, S["train"], R, L=Ls.get(name))
        sync(model)
        fit_secs = time.time() - tic
        tic = time.time()
        rec = dict(method=name, r=R, **evaluate(model, S),
                   n_active=n_active(model),
                   n_active_by_degree=active_by_degree(model),
                   certificate=getattr(model, "certificate_", None),
                   lambda_=getattr(model, "lambda_", None),
                   secs=round(fit_secs, 3),
                   eval_secs=round(time.time() - tic, 3))
        old = results.get(f"{name}@r={R}")
        results[f"{name}@r={R}"] = rec
        models[name] = model
        msg = (f"  [field] {name:18s} r={R} refit: t_in={rec['test_interp']:.2e} "
               f"t_ex={rec['test_extrap']:.2e}")
        if old is not None:
            msg += (f"  (sweep entry replaced; was "
                    f"{old['test_interp']:.2e}/{old['test_extrap']:.2e})")
        log(msg)
    return models
