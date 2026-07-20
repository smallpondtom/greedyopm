"""Shared experiment harness: method registry, resumable r-sweep, field recon.

Methods compared (the caller provides a train / validation / test split):

  * POD                      -- leading-r linear projection (reference floor)
  * Greedy-QM                -- Schwerdtner & Peherstorfer greedy modes, quadratic monomials
  * Geelen-QM                -- leading-r POD modes + group-sparse quadratic columns
  * Greedy-Fixed-OPM (p=2)   -- greedy modes, ALL orthonormal quadratic features
  * Greedy-OPM (p=2)         -- greedy modes + feature selection (orthonormal quadratic)
  * Greedy-OPM (p=[2,3])     -- greedy modes + feature selection (orthonormal quad+cubic)

Data split (done by the caller, see exp_split.py): TEMPORAL -- the first 50%
of the timesteps train the models, the next 25% (near future) form the
validation set used ONLY to tune the Greedy-OPM hyperparameters
(``gamma_sp``, ``mu``), and the final 25% (far future) are the held-out test
set for the reported errors.  Test is therefore a genuine forward-in-time
out-of-distribution extrapolation.

(gamma, gamma_sp) tuning (fit-once-then-resolve).  A GreedyOPM is fit ONCE
per mode-selection variant under the experiment's Lipschitz budget L
(``cfg['budget']``).  Because eligibility, the budgeted knapsack admission,
and the one-step lifting for any ``(gamma, gamma_sp)`` are a cheap
re-selection from the stored full-basis correlation and cached per-feature
gradient bounds (``GreedyOPM.set_sparsity``), the whole grid is swept on the
validation window with a single fit.  Every grid point carries the
certificate ``L_Gamma(delta) <= L`` by construction of the admission.

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
from greedyopm.greedy import _admit                         # noqa: E402
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


_OPM_METHODS = ("Greedy-OPM (p=2)", "Greedy-OPM (p=[2,3])",
                "Greedy-OPM (p=[2,3,4])")
_OPM_DEGREES = {"Greedy-OPM (p=2)": [2], "Greedy-OPM (p=[2,3])": [2, 3],
                "Greedy-OPM (p=[2,3,4])": [2, 3, 4]}


_UNSET = object()


def _fit_opm_base(S_train, r, degrees, cfg, *, s_pool=None, svd=_UNSET):
    """Fit GreedyOPM once at gamma_sp=0 under the experiment's budget.

    ``s_pool=r`` restricts the candidate pool to the leading-r singular
    vectors (the gamma_sp -> infinity limit of the in-loop greedy scoring,
    where the polynomial gains vanish and selection reduces to the singular
    values); the default uses the full pool of ``cfg['s_pool']`` candidates.
    """
    budget = cfg.get("budget")
    if budget is not None and not np.isfinite(budget):
        budget = None
    if s_pool is None:
        # Quartic dictionaries make the per-candidate Arnoldi rebuilds
        # O(n m^2) with m = O(r^4); a reduced candidate pool keeps the
        # greedy variant tractable (the leading variant is unaffected).
        s_pool = (cfg.get("s_pool_p4", 24) if max(degrees) >= 4
                  else cfg["s_pool"])
    return GreedyOPM(
        r=r, s=s_pool, p=degrees,
        gamma_sp=0.0, gamma=cfg["gamma_opm"], ortho_method="arnoldi",
        center=True, n_reorth=2, budget=budget,
        grad_dilation=cfg.get("grad_dilation", 0.1),
        n_grad_samples=cfg.get("n_grad_samples"),
    ).fit(S_train, svd=(cfg.get("_svd") if svd is _UNSET else svd))


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



def _gamma_grid(cfg, n):
    """Ridge (gamma) grid: the base value plus optional multiples of n."""
    base = float(cfg["gamma_opm"])
    factors = cfg.get("gamma_factors")
    if not factors:
        return [base]
    return [base] + [float(f) * n for f in factors]


def _select_config(curves, tol):
    """Pick the best curve entry from the stacked validation curves.

    Validation error is the primary criterion: among all grid points within
    ``tol`` of the best validation error, prefer the lowest TRAINING error
    (richer feature sets are kept whenever they cost nothing on validation);
    break ties by the fewest active features, then the largest
    ``gamma_sp`` (the sparsest-threshold model at equal everything else).
    """
    flat = [c for curve in curves for c in curve]
    if not flat:
        return None
    errs = np.array([c["val_err"] for c in flat])
    acts = np.array([c["n_active"] for c in flat])
    trs = np.array([c.get("train_err", np.inf) for c in flat])
    gsps = np.array([c["gamma_sp"] for c in flat])
    emin = float(errs.min())
    ok = np.where(errs <= (1.0 + tol) * emin)[0]
    # within the validation band: best TRAINING error first (richer feature
    # sets are kept whenever they cost nothing on validation), ...
    tr_min = float(trs[ok].min())
    ok2 = ok[trs[ok] <= (1.0 + 1e-9) * tr_min] if np.isfinite(tr_min) else ok
    # ... then fewest active features, breaking ties by the largest gamma_sp
    order = ok2[np.lexsort((-gsps[ok2], acts[ok2]))]
    return flat[int(order[0])]


def _select_variant(per_variant, variant_tol):
    """Choose between the mode-selection variants (Occam preference).

    ``per_variant`` maps sel -> (model, curves, best_entry).  The simpler
    ``leading`` variant is preferred whenever its tuned validation error is
    within ``variant_tol`` of the tuned greedy variant; otherwise the greedy
    variant (which must then improve validation by more than the margin)
    wins.
    """
    g = per_variant.get("greedy")
    l = per_variant.get("leading")
    if g is None or g[2] is None:
        return "leading"
    if l is None or l[2] is None:
        return "greedy"
    if l[2]["val_err"] <= (1.0 + variant_tol) * g[2]["val_err"]:
        return "leading"
    return "greedy"


def _grid_curves(m, S_train, S_val, cfg, n_points, sel):
    """(gamma, gamma_sp) tuning grid under the model's Lipschitz budget.

    For every grid point, eligibility is the group soft-threshold of the
    one-step theorem, the ACTIVE set is the density-greedy knapsack
    admission under the budget slack ``sqrt(L^2 - 1)`` (all eligible
    features when the budget is None), and the lifting is the unshrunk
    one-step solution ``d_k = soft_k * n / (n + gamma)``.  All evaluations
    run in reduced space from precomputed Gram matrices; every entry
    carries its certificate

        cert = sqrt(1 + (sum_A d_k ||r_k|| L_psi_k)^2) <= L.

    Errors use the reverse-triangle lower bound as a floor, which guards
    against catastrophic cancellation when the features extrapolate
    violently.  Returns a list of gamma_sp-curves, one per gamma.
    """
    n = int(m.Psi_.shape[1])
    S_val = jnp.asarray(S_val)
    S_train = jnp.asarray(S_train)
    # ||S_c||^2 and E2 = ||(I - U U^T) S_c||^2 without materializing the
    # centered (d, n) matrix (memory-lean).
    Sc2 = float(jnp.sum(S_train ** 2)
                + n * jnp.sum(m.s_ref_ ** 2)
                - 2.0 * jnp.sum(m.s_ref_[:, 0] * jnp.sum(S_train, axis=1)))
    S_norm = float(np.sqrt(max(Sc2, 0.0)))
    UtSc = m.U_r_.T @ S_train - (m.U_r_.T @ m.s_ref_)
    E2 = max(Sc2 - float(jnp.sum(UtSc ** 2)), 0.0)
    Sh_val = m.encode(S_val)
    P_val = m._evaluate_orthonormal_features(Sh_val)[m.n_linear_:]
    R0 = S_val - m.s_ref_ - m.U_r_ @ Sh_val            # linear-only residual
    R0n2 = float(jnp.sum(R0 ** 2))
    denom = float(jnp.linalg.norm(S_val - m.s_ref_, "fro"))
    CtC = np.asarray(m.C_nl_.T @ m.C_nl_)              # (m_nl, m_nl)
    CtM1 = np.asarray(m.C_nl_.T @ (R0 @ P_val.T))      # (m_nl, m_nl)
    G_val = np.asarray(P_val @ P_val.T)                # (m_nl, m_nl)
    m._ensure_L_psi()
    L_psi = np.asarray(m.L_psi_)
    norms = np.sqrt(np.asarray(m.importances_))
    degs = m.feature_degrees_
    L = m.budget
    slack = float(np.sqrt(L ** 2 - 1.0)) if L is not None else np.inf

    curves = []
    for gam in _gamma_grid(cfg, n):
        curve = []
        for g in _gamma_sp_grid(m, n_points):
            excess = np.maximum(n * norms - g, 0.0)
            gains = excess ** 2 / (n + gam)
            costs = excess * L_psi / (n + gam)
            a = _admit(gains, costs, excess > 0.0, slack)
            by_deg = {str(int(q)): int(np.count_nonzero(degs[a] == q))
                      for q in m.degrees_}
            if a.size == 0:
                curve.append(dict(
                    gamma_sp=float(g), gamma=float(gam), sel=sel,
                    cert=1.0, n_active=0, by_degree=by_deg,
                    val_err=float(np.sqrt(R0n2)) / denom,
                    train_err=float(np.sqrt(E2)) / S_norm))
                continue
            soft = np.maximum(1.0 - g / (n * norms[a]), 0.0)
            d = soft * n / (n + gam)
            cert = float(np.sqrt(1.0 + np.sum(d * norms[a] * L_psi[a]) ** 2))
            CtC_a = CtC[np.ix_(a, a)]
            cross_v = float(np.sum(np.diag(CtM1[np.ix_(a, a)]) * d))
            X = (d[:, None] * CtC_a) * d[None, :]
            quad_v = float(np.sum(X * G_val[np.ix_(a, a)].T))
            e2v = max(R0n2 - 2.0 * cross_v + quad_v,
                      (np.sqrt(max(quad_v, 0.0)) - np.sqrt(R0n2)) ** 2, 0.0)
            cross_t = float(np.sum(np.diag(CtC_a) * d))
            trX = float(np.einsum("i,ii,i->", d, CtC_a, d))
            e2t = max(E2 - 2.0 * n * cross_t + n * trX,
                      (np.sqrt(max(n * trX, 0.0)) - np.sqrt(E2)) ** 2, 0.0)
            curve.append(dict(
                gamma_sp=float(g), gamma=float(gam), sel=sel,
                cert=cert, n_active=int(a.size), by_degree=by_deg,
                val_err=float(np.sqrt(e2v)) / denom,
                train_err=float(np.sqrt(e2t)) / S_norm))
        curves.append(curve)
    return curves


def fit_opm_tuned(S_train, S_val, r, degrees, cfg):
    """Fit GreedyOPM (two mode-selection candidates), tune on validation.

    Two greedy fits are produced under the experiment's Lipschitz budget:
    the standard greedy mode selection over the full candidate pool
    (``sel='greedy'``) and the leading-r modes (``sel='leading'``).  For
    each fit, the ``(gamma, gamma_sp)`` grid is evaluated in REDUCED space
    (:func:`_grid_curves`); every grid point is a certified configuration.
    Validation picks the variant, ``gamma``, and ``gamma_sp`` jointly.
    Returns ``(model, best_gamma_sp, curve, sel, gamma)``.
    """
    n_points = cfg.get("gamma_sp_points", 24)
    tol = cfg.get("gamma_sp_tol", 0.02)
    per_variant = {}
    for sel, s_pool in (("greedy", None), ("leading", r)):
        m = _fit_opm_base(S_train, r, degrees, cfg, s_pool=s_pool)
        curves = _grid_curves(m, S_train, S_val, cfg, n_points, sel)
        per_variant[sel] = (m, curves, _select_config(curves, tol))

    best_sel = _select_variant(per_variant,
                               cfg.get("variant_tol", tol))
    m, curves, best = per_variant[best_sel]
    per_variant.clear()                       # free the non-selected model
    curve = next(c for c in curves
                 if c and c[0].get("gamma") == best.get("gamma"))
    _apply_opm_params(m, best["gamma_sp"], gamma=best.get("gamma"))
    return m, best["gamma_sp"], curve, best_sel, best.get("gamma")


def _apply_opm_params(m, gamma_sp, gamma=None):
    """Set (gamma_sp[, gamma]) on a fitted GreedyOPM and re-run admission."""
    if gamma is not None:
        m.gamma = float(gamma)
    m.set_sparsity(float(gamma_sp))
    return m


def _opm_model(S_train, r, degrees, cfg, gamma_sp, sel="greedy",
               gamma=None, param_scale=1.0):
    """Build a pruned GreedyOPM for KNOWN (gamma_sp, sel, gamma).

    ``param_scale`` rescales the stored scale-dependent parameters when the
    model is rebuilt on a different number of snapshots than the tuning fit;
    pass ``n_new / n_tuning``.  When ``gamma_sp is None`` (legacy callers
    with no validation set) fall back to the ``tau`` heuristic.
    """
    m = _fit_opm_base(S_train, r, degrees, cfg,
                      s_pool=(r if sel == "leading" else None))
    if gamma_sp is None:
        imp = np.asarray(m.importances_)
        n = int(S_train.shape[1])
        gamma_sp = (float(n * np.sqrt(cfg.get("tau", 1e-6) * imp.max()))
                    if imp.size else 0.0)
    return _apply_opm_params(
        m, gamma_sp * param_scale,
        gamma=(None if gamma is None else gamma * param_scale))


# ----------------------------------------------------------------------------
# model builders  (name -> object with .reconstruct / .reconstruction_error)
# ----------------------------------------------------------------------------

def build_model(name, S_train, r, cfg, *, gamma_sp=None,
                sel="greedy", gamma=None, param_scale=1.0):
    """Build a single fitted model.

    ``gamma_sp`` / ``sel`` / ``gamma`` (Greedy-OPM variants only) set the
    sparsity threshold, mode-selection variant, and ridge weight to KNOWN,
    previously validation-tuned values; when ``gamma_sp is None`` the
    ``tau`` heuristic is used instead.  Ignored for all non-OPM methods.
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
        return _opm_model(S_train, r, _OPM_DEGREES[name], cfg, gamma_sp,
                          sel or "greedy", gamma, param_scale)
    raise ValueError(name)


METHOD_NAMES = [
    "POD",
    "Greedy-QM",
    "Geelen-QM",
    "Greedy-Fixed-OPM (p=2)",
    "Greedy-OPM (p=2)",
    "Greedy-OPM (p=[2,3])",
    "Greedy-OPM (p=[2,3,4])",
]


def opm_feasible(name, r, n_train):
    """Unisolvence guard: empirical orthonormality needs n >= #features."""
    if name not in _OPM_METHODS:
        return True
    from math import comb
    degs = _OPM_DEGREES[name]
    m_total = comb(r + max(degs), max(degs))       # all grlex features <= p
    return n_train >= m_total

METHOD_STYLES = {
    "POD":                    dict(fmt="-o",  color="black",   lw=3, ms=8),
    "Greedy-QM":              dict(fmt="--^", color="#1f77b4", lw=2.5, ms=8),
    "Geelen-QM":              dict(fmt="--v", color="#ff7f0e", lw=2.5, ms=8),
    "Greedy-Fixed-OPM (p=2)": dict(fmt="-s",  color="#2ca02c", lw=2.5, ms=8),
    "Greedy-OPM (p=2)":       dict(fmt="-P",  color="#9467bd", lw=2.5, ms=9),
    "Greedy-OPM (p=[2,3])":   dict(fmt="-D",  color="#d62728", lw=2.5, ms=8),
    "Greedy-OPM (p=[2,3,4])": dict(fmt="-X",  color="#8c564b", lw=2.5, ms=9),
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
    n_train = int(S_train.shape[1])
    for r in rs:
        for name in methods:
            key = f"{name}@r={r}"
            if key in results:
                continue
            if not opm_feasible(name, r, n_train):
                continue                     # dictionary exceeds n_train
            if time.time() - t0 > time_budget and done_any:
                _save(ckpt_path, results)
                print(f"[budget] stopping after {time.time()-t0:.1f}s; "
                      f"{_remaining(results, rs, methods)} entries left")
                return results, False
            tic = time.time()
            if name in _OPM_METHODS:
                model, gsp, curve, sel, gam = fit_opm_tuned(
                    S_train, S_val, r, _OPM_DEGREES[name], cfg)
            else:
                model, gsp, curve, sel, gam = (
                    build_model(name, S_train, r, cfg), None, None,
                    None, None)
            fit_secs = round(time.time() - tic, 3)     # fit + tuning wall time
            etr = model.reconstruction_error(S_train, relative=True)
            eva = model.reconstruction_error(S_val, relative=True)
            ete = model.reconstruction_error(S_test, relative=True)
            results[key] = dict(method=name, r=int(r), train=etr, val=eva,
                                test=ete, n_active=_n_active(model),
                                n_active_by_degree=_active_by_degree(model),
                                gamma_sp=(None if gsp is None else float(gsp)),
                                gamma=(None if gam is None else float(gam)),
                                cert=(None if getattr(model, "certificate_", None)
                                      is None else float(model.certificate_)),
                                sel=sel,
                                gamma_sp_curve=curve,
                                fit_secs=fit_secs,
                                secs=round(time.time() - tic, 2))
            done_any = True
            cert = results[key].get("cert")
            gtxt = (f" gsp={gsp:.2e} cert={cert:.3g} sel={sel}"
                    if gsp is not None else "")
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
