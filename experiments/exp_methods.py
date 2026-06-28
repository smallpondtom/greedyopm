"""Shared experiment harness: method registry, resumable r-sweep, field recon.

Methods compared (all share the even/odd train/test split done by the caller):

  * POD                      -- leading-r linear projection (reference floor)
  * Greedy-QM                -- Schwerdtner & Peherstorfer greedy modes, quadratic monomials
  * Geelen-QM                -- leading-r POD modes + group-sparse quadratic columns
  * Greedy-Fixed-OPM (p=2)   -- greedy modes, ALL orthonormal quadratic features
  * Greedy-OPM (p=2)         -- greedy modes + feature selection (orthonormal quadratic)
  * Greedy-OPM (p=[2,3])     -- greedy modes + feature selection (orthonormal quad+cubic)

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


def _opm_select(S_train, r, s_pool, degrees, gamma, tau):
    """Greedy-OPM with adaptive feature selection.

    Fit once with gamma_sp=0 to read feature importances, then set the
    sparsity threshold to retain features whose individual gain exceeds
    ``tau`` times the largest gain, and refit with that threshold.
    """
    n = S_train.shape[1]
    g0 = GreedyOPM(r=r, s=s_pool, p=degrees, gamma_sp=0.0, gamma=gamma,
                   ortho_method="arnoldi", center=True, n_reorth=2).fit(S_train)
    imp = np.asarray(g0.importances_)
    gsp = float(n * np.sqrt(tau * imp.max())) if imp.size else 0.0
    print(f"    OPM feature selection: gamma_sp={gsp:.1e} (tau={tau})")
    m = GreedyOPM(r=r, s=s_pool, p=degrees, gamma_sp=gsp, gamma=gamma,
                  ortho_method="arnoldi", center=True, n_reorth=2).fit(S_train)
    return m


# ----------------------------------------------------------------------------
# model builders  (name -> object with .reconstruct / .reconstruction_error)
# ----------------------------------------------------------------------------

def build_model(name, S_train, r, cfg):
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
    if name == "Greedy-OPM (p=2)":
        return _opm_select(S_train, r, s_pool, [2], cfg["gamma_opm"], cfg["tau"])
    if name == "Greedy-OPM (p=[2,3])":
        return _opm_select(S_train, r, s_pool, [2, 3], cfg["gamma_opm"], cfg["tau"])
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


# ----------------------------------------------------------------------------
# resumable sweep
# ----------------------------------------------------------------------------

def run_sweep(S_train, S_test, rs, cfg, ckpt_path, time_budget=38.0,
              methods=None):
    methods = methods or METHOD_NAMES
    results = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            results = json.load(f)

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
            model = build_model(name, S_train, r, cfg)
            fit_secs = round(time.time() - tic, 3)     # fit-only wall time
            etr = model.reconstruction_error(S_train, relative=True)
            ete = model.reconstruction_error(S_test, relative=True)
            results[key] = dict(method=name, r=int(r), train=etr, test=ete,
                                n_active=_n_active(model),
                                fit_secs=fit_secs,
                                secs=round(time.time() - tic, 2))
            done_any = True
            print(f"  r={r:2d}  {name:24s} train={etr:.3e} test={ete:.3e} "
                  f"act={results[key]['n_active']:>3d}  ({results[key]['secs']}s)")
            _save(ckpt_path, results)
    _save(ckpt_path, results)
    return results, True


def _remaining(results, rs, methods):
    return sum(1 for r in rs for m in methods if f"{m}@r={r}" not in results)


def _save(path, results):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=0)
