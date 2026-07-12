"""Shared experiment harness: method registry, resumable r-sweep, field recon.

Methods compared (the caller provides a train / validation / test split):

  * POD                      -- leading-r linear projection (reference floor)
  * Greedy-QM                -- Schwerdtner & Peherstorfer greedy modes, quadratic monomials
  * Geelen-QM                -- leading-r POD modes + group-sparse quadratic columns
  * Greedy-Fixed-OPM (p=2)   -- greedy modes, ALL orthonormal quadratic features
  * Greedy-OPM (p=2)         -- greedy modes + feature selection (orthonormal quadratic)
  * Greedy-OPM (p=[2,3])     -- greedy modes + feature selection (orthonormal quad+cubic)

Data split (done by the caller): the first half of the snapshots is used for
training; the remaining half is split into a validation set (used ONLY to tune
the group-sparsity parameter ``gamma_sp`` of the two Greedy-OPM variants) and a
held-out test set (used for the reported errors).

gamma_sp tuning (fit-once-then-prune).  A GreedyOPM is fit ONCE at
``gamma_sp=0`` (all nonlinear features active -- the richest greedy mode
selection).  Because the active set and debiased lifting for any ``gamma_sp``
are a cheap re-thresholding of the stored full-basis correlation
(``GreedyOPM.set_sparsity``), the whole ``gamma_sp`` grid is swept on the
validation set with a single fit.  The selected value is the SPARSEST
``gamma_sp`` whose validation reconstruction error is within ``gamma_sp_tol``
of the best -- balancing reconstruction accuracy and feature reduction.

The sweep checkpoints every (method, r) to a JSON file and skips finished
entries, and it stops early once a wall-clock budget is hit, so it can be run
repeatedly under the 45s-per-call sandbox limit until complete.
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
import jax.numpy as jnp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # qm/, data/
sys.path.insert(0, str(_HERE.parent / "src"))        # greedyopm (sandbox)

from greedyopm import GreedyOPM, GreedyFixedOPM             # noqa: E402
from qm.quadmani import quadmani_greedy, quadratic_mapping  # noqa: E402
from qm.geelen_qm import GeelenQM                           # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def relative_error(S_true, S_approx):
    return float(jnp.linalg.norm(S_true - S_approx, "fro")
                 / jnp.linalg.norm(S_true, "fro"))


class _PODModel:
    def __init__(self, U_r, s_ref):
        self.U_r, self.s_ref = U_r, s_ref

    def reconstruct(self, S):
        S = jnp.asarray(S)
        return self.s_ref + self.U_r @ (self.U_r.T @ (S - self.s_ref))

    def reconstruction_error(self, S, relative=True):
        S = jnp.asarray(S)
        e = float(jnp.linalg.norm(S - self.reconstruct(S), "fro"))
        return e / float(jnp.linalg.norm(S - self.s_ref, "fro")) if relative else e


class _QMModel:
    """Wrap Greedy-QM functional output with a reconstruct() API."""
    def __init__(self, V, Vbar, mu, feature_map):
        self.V, self.Vbar = V, Vbar
        self.mu = mu.reshape(-1, 1)
        self.fmap = feature_map

    def reconstruct(self, S):
        S = jnp.asarray(S)
        Sh = self.V.T @ (S - self.mu)
        return self.mu + self.V @ Sh + self.Vbar @ self.fmap(Sh)

    def reconstruction_error(self, S, relative=True):
        S = jnp.asarray(S)
        e = float(jnp.linalg.norm(S - self.reconstruct(S), "fro"))
        return e / float(jnp.linalg.norm(S - self.mu, "fro")) if relative else e


_OPM_METHODS = ("Greedy-OPM (p=2)", "Greedy-OPM (p=[2,3])")
_OPM_DEGREES = {"Greedy-OPM (p=2)": [2], "Greedy-OPM (p=[2,3])": [2, 3]}


def _fit_opm_base(S_train, r, degrees, cfg):
    """Fit GreedyOPM once at gamma_sp=0 (all nonlinear features active)."""
    return GreedyOPM(
        r=r, s=cfg["s_pool"], p=degrees, gamma_sp=0.0,
        gamma=cfg["gamma_opm"], ortho_method="arnoldi",
        center=True, n_reorth=2,
    ).fit(S_train)


def _gamma_sp_grid(base_model, n_points):
    """Log-spaced gamma_sp grid spanning all-active .. all-pruned for this fit.

    A nonlinear feature k is pruned once ``gamma_sp/n >= sqrt(importance_k)``,
    so the active set only changes for ``gamma_sp in [n*min_gain, n*max_gain]``.
    We bracket that range (plus a 0 point for the all-active plateau).
    """
    n = int(base_model.Psi_.shape[1])
    norms = np.sqrt(np.asarray(base_model.importances_))
    norms = norms[norms > 0.0]
    if norms.size == 0:
        return np.array([0.0])
    lo = 0.5 * float(n * norms.min())
    hi = 1.5 * float(n * norms.max())
    grid = np.logspace(np.log10(lo), np.log10(hi), int(n_points))
    return np.unique(np.concatenate([[0.0], grid]))


def _gamma_sp_curve(base_model, S_val, n_points):
    """Validation reconstruction error / active-feature counts vs gamma_sp.

    Cheap: reuses the single greedy fit, re-thresholding via ``set_sparsity``.
    """
    S_val = jnp.asarray(S_val)
    curve = []
    for g in _gamma_sp_grid(base_model, n_points):
        base_model.set_sparsity(float(g))
        curve.append(dict(
            gamma_sp=float(g),
            val_err=base_model.reconstruction_error(S_val, relative=True),
            n_active=int(base_model.n_selected_nonlinear),
            by_degree={str(k): int(v)
                       for k, v in base_model.active_degree_counts.items()},
        ))
    return curve


def _select_gamma_sp(curve, tol):
    """Sparsest gamma_sp whose validation error is within ``tol`` of the best."""
    if not curve:
        return 0.0
    gs = np.array([c["gamma_sp"] for c in curve])
    errs = np.array([c["val_err"] for c in curve])
    emin = float(errs.min())
    ok = np.where(errs <= (1.0 + tol) * emin)[0]
    best = int(ok[np.argmax(gs[ok])])          # largest gamma_sp within tol
    return float(gs[best])


def fit_opm_tuned(S_train, S_val, r, degrees, cfg):
    """Fit GreedyOPM once, tune gamma_sp on the validation set, prune to best.

    Returns ``(model, best_gamma_sp, curve)`` where ``curve`` is the full
    validation error / feature-count sweep used for the gamma_sp study plot.
    """
    m = _fit_opm_base(S_train, r, degrees, cfg)
    curve = _gamma_sp_curve(m, S_val, cfg.get("gamma_sp_points", 24))
    best = _select_gamma_sp(curve, cfg.get("gamma_sp_tol", 0.02))
    m.set_sparsity(best)
    return m, best, curve


def _opm_model(S_train, r, degrees, cfg, gamma_sp):
    """Build a pruned GreedyOPM for a KNOWN gamma_sp (field / timeseries recon).

    When ``gamma_sp is None`` (legacy callers with no validation set) fall back
    to the ``tau`` heuristic so those scripts keep working unchanged.
    """
    m = _fit_opm_base(S_train, r, degrees, cfg)
    if gamma_sp is None:
        imp = np.asarray(m.importances_)
        n = int(S_train.shape[1])
        gamma_sp = (float(n * np.sqrt(cfg.get("tau", 1e-6) * imp.max()))
                    if imp.size else 0.0)
    m.set_sparsity(float(gamma_sp))
    return m


# ----------------------------------------------------------------------------
# model builders  (name -> object with .reconstruct / .reconstruction_error)
# ----------------------------------------------------------------------------

def build_model(name, S_train, r, cfg, *, gamma_sp=None):
    """Build a single fitted model.

    ``gamma_sp`` (Greedy-OPM variants only) sets the sparsity threshold to a
    KNOWN, previously validation-tuned value; when ``None`` the ``tau``
    heuristic is used instead.  Ignored for all non-OPM methods.
    """
    S_train = jnp.asarray(S_train)
    s_pool = cfg["s_pool"]
    if name == "POD":
        s_ref = jnp.mean(S_train, axis=1, keepdims=True)
        U, _, _ = jnp.linalg.svd(S_train - s_ref, full_matrices=False)
        return _PODModel(U[:, :r], s_ref)
    if name == "Greedy-QM":
        V, Vbar, mu, _ = quadmani_greedy(
            S_train, r=r, n_vectors_to_check=s_pool,
            reg_magnitude=cfg["gamma_qm"], idx_in=np.array([], dtype=int),
            feature_map=quadratic_mapping, shifting=True)
        return _QMModel(V, Vbar, mu, quadratic_mapping)
    if name == "Geelen-QM":
        return GeelenQM(r=r, gamma_sp_rel=cfg["geelen_rel"],
                        gamma=cfg["gamma_geelen"], n_iter=cfg["geelen_iters"]).fit(S_train)
    if name == "Greedy-Fixed-OPM (p=2)":
        return GreedyFixedOPM(r=r, s=s_pool, p=[2], gamma=cfg["gamma_opm"],
                              ortho_method="arnoldi", center=True).fit(S_train)
    if name in _OPM_METHODS:
        return _opm_model(S_train, r, _OPM_DEGREES[name], cfg, gamma_sp)
    raise ValueError(name)


METHOD_NAMES = [
    "POD",
    "Greedy-QM",
    "Geelen-QM",
    "Greedy-Fixed-OPM (p=2)",
    "Greedy-OPM (p=2)",
    "Greedy-OPM (p=[2,3])",
]

METHOD_STYLES = {
    "POD":                    dict(fmt="-o",  color="black",   lw=3, ms=8),
    "Greedy-QM":              dict(fmt="--^", color="#1f77b4", lw=2.5, ms=8),
    "Geelen-QM":              dict(fmt="--v", color="#ff7f0e", lw=2.5, ms=8),
    "Greedy-Fixed-OPM (p=2)": dict(fmt="-s",  color="#2ca02c", lw=2.5, ms=8),
    "Greedy-OPM (p=2)":       dict(fmt="-P",  color="#9467bd", lw=2.5, ms=9),
    "Greedy-OPM (p=[2,3])":   dict(fmt="-D",  color="#d62728", lw=2.5, ms=8),
}


def _n_active(model):
    for attr in ("n_selected_nonlinear", "n_active"):
        if hasattr(model, attr):
            return int(getattr(model, attr))
    if hasattr(model, "active_nl_global_"):
        return int(np.asarray(model.active_nl_global_).size)
    return -1


def _active_by_degree(model):
    """`{degree: count}` breakdown of the active nonlinear features, else None.

    Only the OPM models expose ``active_degree_counts`` (e.g. quad vs cubic for
    ``p=[2,3]``); other methods return ``None``.
    """
    if hasattr(model, "active_degree_counts"):
        return {str(k): int(v) for k, v in model.active_degree_counts.items()}
    return None


# ----------------------------------------------------------------------------
# resumable sweep
# ----------------------------------------------------------------------------

def run_sweep(S_train, S_val, S_test, rs, cfg, ckpt_path, time_budget=38.0,
              methods=None):
    """Resumable (method, r) sweep with per-(method, r) gamma_sp tuning.

    For the two Greedy-OPM variants the model is fit once and ``gamma_sp`` is
    tuned on ``S_val`` (see :func:`fit_opm_tuned`); the full validation curve is
    stored under ``gamma_sp_curve`` for the study plot.  Reported ``train`` /
    ``test`` errors use the held-out splits; ``val`` is stored for reference.
    """
    methods = methods or METHOD_NAMES
    results = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            results = json.load(f)

    S_train = jnp.asarray(S_train)
    S_val = jnp.asarray(S_val)
    S_test = jnp.asarray(S_test)

    t0 = time.time()
    done_any = False
    for r in rs:
        for name in methods:
            key = f"{name}@r={r}"
            if key in results:
                continue
            if time.time() - t0 > time_budget and done_any:
                _save(ckpt_path, results)
                print(f"[budget] stopping after {time.time()-t0:.1f}s; "
                      f"{_remaining(results, rs, methods)} entries left")
                return results, False
            tic = time.time()
            if name in _OPM_METHODS:
                model, gsp, curve = fit_opm_tuned(
                    S_train, S_val, r, _OPM_DEGREES[name], cfg)
            else:
                model, gsp, curve = build_model(name, S_train, r, cfg), None, None
            fit_secs = round(time.time() - tic, 3)     # fit + tuning wall time
            etr = model.reconstruction_error(S_train, relative=True)
            eva = model.reconstruction_error(S_val, relative=True)
            ete = model.reconstruction_error(S_test, relative=True)
            results[key] = dict(method=name, r=int(r), train=etr, val=eva,
                                test=ete, n_active=_n_active(model),
                                n_active_by_degree=_active_by_degree(model),
                                gamma_sp=(None if gsp is None else float(gsp)),
                                gamma_sp_curve=curve,
                                fit_secs=fit_secs,
                                secs=round(time.time() - tic, 2))
            done_any = True
            gtxt = f" gsp={gsp:.2e}" if gsp is not None else ""
            print(f"  r={r:2d}  {name:24s} train={etr:.3e} test={ete:.3e} "
                  f"act={results[key]['n_active']:>3d}{gtxt}  "
                  f"({results[key]['secs']}s)")
            _save(ckpt_path, results)
    _save(ckpt_path, results)
    return results, True


def _remaining(results, rs, methods):
    return sum(1 for r in rs for m in methods if f"{m}@r={r}" not in results)


def _save(path, results):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=0)
