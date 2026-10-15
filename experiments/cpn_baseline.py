"""CPN-S baseline: the compositional polynomial network of Bensalah, Nouy &
Soffo (2026), run through their released code (``third_party/CPN_MOR``).

The released code is used *as is* (same hyperbolic-cross Legendre features,
same LARS + leave-one-out sparse least squares via ``tensap``, same adaptive
acceptance rule with the default weights ``alpha = 1``, ``beta = 1/sqrt 2``
and the mean-squared setting).  Only two things are decided here, and both
are needed to put CPN-S on the same footing as the other methods:

* the background space ``X_N`` is spanned by the leading ``N`` POD modes of
  the (centered) training snapshots -- ``N`` is passed in and set equal to the
  candidate pool ``s`` of the greedy methods, instead of being chosen by the
  tolerance-based truncation of the released driver;
* the greedy loop of ``find_n`` may be *capped*: if the adaptive rule would
  promote more than ``n_max`` modes to the linear part the run is abandoned
  (returns ``None``), which only saves time in a tolerance sweep whose
  interesting runs are those with ``n <= R_MAX``.  ``_find_n`` below is a
  verbatim copy of ``CPN_S.find_n`` plus that cap and without the prints.

CPN-S does not take the manifold dimension as an input: the dimension
``n = |I|`` is an *outcome* of the target tolerance ``tol``.  The driver
therefore sweeps ``tol`` and records, per achieved ``n``, the run with the
tightest tolerance (see :func:`sweep_tolerances`).

The Lipschitz target ``L`` (``Gamma`` in their code) is accepted for
completeness only: in the released code the Lipschitz check in ``find_n`` is
commented out, so it has no effect on the fit.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

_CPN = Path(__file__).resolve().parent / "third_party" / "CPN_MOR"
for _p in (str(_CPN), str(_CPN / "train_utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tensap  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from train_utils.utilities_sparse import CPN_S  # noqa: E402
from train_utils.utilities import lp_error  # noqa: E402
from train_utils.solve_ls import solve_ls  # noqa: E402
from encoder_decoder import Decoder, Encoder  # noqa: E402

CPN_COMMIT = "74e2228b2c64dc601fb32dff353aefbf264f169d"  # vendored revision


class _Capped(Exception):
    """Raised inside the greedy loop when ``n`` would exceed ``n_max``."""


def _find_n(method: CPN_S, U, Q, p1, *, n_min=1, train_set=1.0, Gamma=100,
            n_max=None, index_set="hyperbolic_cross"):
    """``CPN_S.find_n`` of the released code, verbatim except for

    * the ``n_max`` cap (abandon the run instead of promoting an
      ``(n_max + 1)``-th mode),
    * the choice of the polynomial index set: ``"hyperbolic_cross"`` is the
      released default (``prod(nu_k + 1) <= p + 1``: univariate powers up to
      ``p`` and bilinear cross terms only); ``"total_degree"`` is the full
      total-degree set ``|nu|_1 <= p`` (the line their code keeps commented
      out), which gives CPN-S the same degree-``p`` dictionary as the other
      cubic methods, and
    * the console prints removed.
    """
    self = method
    coeffs_dict = {}
    func = {}
    N_train = int(Q.shape[1] * train_set)
    N_val = Q.shape[1] - N_train
    for j, q_j in enumerate(Q):
        coeffs_dict[f"coef{j}"] = {'value': q_j, 'index': j}

    n = n_min
    index_r = [r for r in range(n)]
    for k in index_r:
        coeffs_dict.pop(f"coef{k}")
    indices_list = [value["index"] for value in coeffs_dict.values()]
    dim = n

    Qr = Q[:n, :]
    Q_check = Q[:n, :]

    learnt_weights = []
    learnt_weights_gamma = []
    lipschitz_consts = []
    lip_consts_inputs = []

    ls = tensap.LinearModelLearningSquareLoss()
    ls.regularization = True
    ls.regularization_type = "l1"
    ls.regularization_options = {"alpha": 0.}
    ls.model_selection = True
    if ls.regularization_type == "l2":
        ls.model_selection = False
    ls.error_estimation = True
    ls.error_estimation_type = 'leave_out'
    ls.error_estimation_options["correction"] = True
    while indices_list:  # step
        learnt_w = []
        learnt_w_gamma = []
        lip_consts = []
        X = [tensap.UniformRandomVariable(np.min(x), np.max(x)) for x in Q_check]
        BASIS = [
            tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(), range(p1 + 1))
            for x in X
        ]
        BASES = tensap.FunctionalBases(BASIS)
        if index_set == "total_degree":
            I = tensap.MultiIndices.with_bounded_norm(dim, 1, p1)
        else:
            I = tensap.MultiIndices.hyperbolic_cross_set(dim, p1)
        s = solve_ls(dim=dim, I=I, maxIndex=p1)
        H = tensap.SparseTensorProductFunctionalBasis(BASES, I)
        A = H.eval(Q_check.T[:N_train, :])
        coeffs = Q[indices_list, :].T
        f = Parallel(n_jobs=-1)(
            delayed(self.sparse_solver)(s, Q_check.T[:N_train, :], coeffs[:N_train, i], A, H, ls)
            for i in range(coeffs.shape[1]))
        pred = np.array([f[i](Q_check.T) for i in range(len(f))]).T
        for (i, j) in zip(indices_list, range(len(indices_list))):
            coeff_name = f'coef{i}'
            g_i = self.gamma_i(i, indices_list, learnt_weights_gamma, Gamma)
            lip_const_i = self.lip_norm_i(Q_check, pred[:, j], index_r,
                                          np.array(lip_consts_inputs))

            if lp_error(coeffs[-N_val:, j], pred[-N_val:, j]) <= self.tol_eps_wise(i, indices_list,
                                                                                   learnt_weights):  # and lip_const_i <= g_i:
                func[coeff_name] = {'function': f[j], 'index': i, 'nb_deps': len(Q_check),
                                    'lip_constant': lip_const_i}
                coeffs_dict.pop(coeff_name)
                w_i = self.new_weights(lp_error(coeffs[-N_val:, j], pred[-N_val:, j]))
                w_i_gamma = self.new_weights_gamma(lip_const_i, Gamma)
                learnt_w.append(w_i)
                learnt_w_gamma.append(w_i_gamma)
                lip_consts.append(lip_const_i)

        learnt_weights.extend(learnt_w)
        learnt_weights_gamma.extend(learnt_w_gamma)
        next_coef = f'coef{dim}'
        if next_coef in coeffs_dict:
            if n_max is not None and len(index_r) + 1 > n_max:
                raise _Capped(len(index_r) + 1)
            q_next = coeffs_dict.pop(next_coef)
            Qr = np.concatenate([Qr, q_next['value'][None,]], axis=0)
            index_r.append(q_next['index'])
            Q_check = np.concatenate([Q_check, q_next['value'][None,]], axis=0)
        else:
            nb_deps = func[next_coef]['nb_deps']
            q_next_approx = func[next_coef]['function'](Q_check.T[:, :nb_deps])
            Q_check = np.concatenate([Q_check, q_next_approx[None,]], axis=0)
            lip_consts_inputs.append(func[next_coef]['lip_constant'])
        lipschitz_consts.extend(lip_consts)
        indices_list = [value['index'] for value in coeffs_dict.values()]
        dim += 1

    return Qr, dict(sorted(func.items(), key=lambda x: x[1]["index"])), index_r, lipschitz_consts


class CPNSparse:
    """CPN-S manifold fitted with the released ``CPN_MOR`` code.

    Arguments
    ---------
    tol : float
        Target (relative, mean-squared) tolerance ``epsilon`` of CPN-S; the
        manifold dimension ``n`` is an outcome of this value.
    N : int
        Dimension of the background space ``X_N`` (leading POD modes).
    p : int
        Polynomial degree of the hyperbolic-cross features (CPN default 3).
    alpha, beta : float
        CPN weight parameters (released defaults ``1`` and ``0.7071``).
    L : float
        CPN Lipschitz target ``Gamma`` (released default 100; inactive in
        the released code, see module docstring).
    n_min : int
        Initial number of linear modes (released default 1).
    train_val_set : float
        Fraction of the training columns used for the fits; the rest is the
        CPN-internal validation set (released default 1.0 = no split).
    n_max : int | None
        Abandon the fit if more than ``n_max`` modes get promoted.
    index_set : str
        ``"hyperbolic_cross"`` (released default) or ``"total_degree"``
        (full degree-``p`` dictionary, see :func:`_find_n`).
    """

    def __init__(self, tol: float, N: int | None = None, *, p: int = 3,
                 alpha: float = 1.0,
                 beta: float = 1.0 / np.sqrt(2.0), L: float = 100.0,
                 n_min: int = 1, train_val_set: float = 1.0,
                 n_max: int | None = None,
                 index_set: str = "hyperbolic_cross"):
        self.tol = float(tol)
        #: ``None`` = the released rule (Section 4.1 of the paper): ``N`` is
        #: the smallest truncation whose PCA tail energy is at most
        #: ``(beta * tol * ||S||_F)^2``, so ``N`` grows as ``tol`` tightens.
        #: An integer fixes ``N`` instead.
        self.N = None if N is None else int(N)
        self.p = int(p)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.L = float(L)
        self.n_min = int(n_min)
        self.train_val_set = float(train_val_set)
        self.n_max = n_max
        if index_set not in ("hyperbolic_cross", "total_degree"):
            raise ValueError(f"Unknown index_set {index_set!r}")
        self.index_set = index_set
        self.capped_ = False

    # ------------------------------------------------------------------
    def fit(self, S) -> "CPNSparse":
        S = np.asarray(S, dtype=np.float64)
        d, m = S.shape
        sref = np.mean(S, axis=1)
        Sref = np.tile(sref[:, None], (1, m))
        self.s_ref_ = sref[:, None]

        U, sig, _ = np.linalg.svd(S - Sref, full_matrices=False)
        method = CPN_S(S=S, Sref=Sref, tol_eps=self.tol, alpha=self.alpha,
                       beta=self.beta, setting="mean_squared")
        if self.N is None:
            # their truncation rule (CPN_S.truncate_svd, prints silenced)
            with contextlib.redirect_stdout(io.StringIO()):
                Vstar, Qstar = method.truncate_svd(U)
            Vstar = np.ascontiguousarray(Vstar)
            N = int(Vstar.shape[1])
        else:
            N = min(self.N, U.shape[1])
            Vstar = np.ascontiguousarray(U[:, :N])
            Qstar = Vstar.T @ (S - Sref)
        self.N_ = N
        self.sigma_ = sig
        self.pod_floor_rel_ = float(np.sqrt(np.sum(sig[N:] ** 2))
                                    / np.linalg.norm(S - Sref, "fro"))
        tic = time.time()
        sink = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                Qr, func, index_r, lip = _find_n(
                    method, Vstar, Qstar, self.p, n_min=self.n_min,
                    train_set=self.train_val_set, Gamma=self.L,
                    n_max=self.n_max, index_set=self.index_set)
        except _Capped as c:
            self.capped_ = True
            self.n_ = int(c.args[0])
            self.fit_secs_ = time.time() - tic
            return self
        self.fit_secs_ = time.time() - tic

        index_r = np.asarray(index_r)
        indices = np.arange(N)
        V = Vstar[:, index_r]
        Vbar = np.delete(Vstar, index_r, axis=1)
        self.encoder_ = Encoder(index=index_r, basis=V, u_ref=sref)
        self.decoder_ = Decoder(linear_index=index_r,
                                nonlinear_index=np.delete(indices, index_r),
                                linear_basis=V, nonlinear_basis=Vbar,
                                functions_f=func, u_ref=sref)
        self.index_r_ = index_r
        self.n_ = int(len(index_r))
        self.r = self.n_
        self.lipschitz_consts_ = [float(x) for x in lip]
        self.cpn_lipschitz_ = float(np.sqrt(1.0 + np.sum(np.asarray(lip) ** 2)))
        self._count_features(func)
        return self

    # ------------------------------------------------------------------
    def _count_features(self, func: dict) -> None:
        """Feature bookkeeping of the sparse tail functions.

        ``n_active_nonlinear`` counts the *distinct* nonlinear (total degree
        >= 2) hyperbolic-cross basis functions with a nonzero coefficient in
        at least one tail function -- the closest analog of an active
        nonlinear feature column of the other methods.  ``n_nonzero_coeffs``
        is the total number of nonzero coefficients over all tail functions
        (constant and linear terms included).
        """
        active = set()
        total_nz = 0
        by_degree: dict[int, int] = {}
        nz_per_function = []
        for rec in func.values():
            f = rec["function"]
            coef = np.asarray(f.data).ravel()
            mi = np.asarray(f.basis.indices.array)
            nz = np.nonzero(coef)[0]
            total_nz += len(nz)
            nz_per_function.append(int(len(nz)))
            for k in nz:
                row = tuple(int(v) for v in mi[k])
                deg = sum(row)
                if deg >= 2:
                    key = tuple(np.trim_zeros(np.asarray(row), "b").tolist())
                    if key not in active:
                        active.add(key)
                        by_degree[deg] = by_degree.get(deg, 0) + 1
        self.active_features_ = active
        self.n_active_nonlinear_ = len(active)
        self.n_nonzero_coeffs_ = int(total_nz)
        self.nz_per_function_ = nz_per_function
        self.active_degree_counts_ = dict(sorted(by_degree.items()))
        self.n_learnt_tail_ = len(func)
        self.nb_deps_ = {int(v["index"]): int(v["nb_deps"]) for v in func.values()}

    # ------------------------------------------------------------------
    @property
    def n_active_nonlinear(self) -> int:
        return int(getattr(self, "n_active_nonlinear_", 0))

    @property
    def active_degree_counts(self) -> dict:
        return dict(getattr(self, "active_degree_counts_", {}))

    def encode(self, S):
        return self.encoder_(np.asarray(S, dtype=np.float64))

    def decode(self, Q):
        out = self.decoder_(np.asarray(Q, dtype=np.float64))
        return out if out.ndim == 2 else out[:, None]

    def reconstruct(self, S):
        return self.decode(self.encode(S))

    def reconstruction_error(self, S, *, relative: bool = True) -> float:
        S = np.asarray(S, dtype=np.float64)
        err = float(np.linalg.norm(S - self.reconstruct(S), "fro"))
        if relative:
            return err / float(np.linalg.norm(S - self.s_ref_, "fro"))
        return err

    def summary(self) -> dict:
        return dict(
            tol=self.tol, N=int(self.N_), N_rule=("fixed" if self.N is not None
                                                 else "beta*tol truncation"),
            p=self.p, alpha=self.alpha, beta=self.beta,
            L=self.L, n_min=self.n_min, train_val_set=self.train_val_set,
            index_set=self.index_set,
            n=int(self.n_), capped=bool(self.capped_),
            fit_secs=round(float(self.fit_secs_), 1),
            pod_floor_rel=self.pod_floor_rel_,
            n_active_nonlinear=self.n_active_nonlinear,
            n_active_by_degree={str(k): int(v)
                                for k, v in self.active_degree_counts.items()},
            n_nonzero_coeffs=int(getattr(self, "n_nonzero_coeffs_", 0)),
            n_learnt_tail=int(getattr(self, "n_learnt_tail_", 0)),
            cpn_lipschitz=float(getattr(self, "cpn_lipschitz_", float("nan"))),
            cpn_commit=CPN_COMMIT,
        )


# ---------------------------------------------------------------------------
# Tolerance sweep
# ---------------------------------------------------------------------------


def sweep_tolerances(build, evaluate, *, n_max: int, tol_hi: float = 0.5,
                     tol_lo: float = 1e-6, per_decade: int = 4,
                     min_ratio: float = 1.03, log=print, prior_runs=None,
                     checkpoint=None) -> dict:
    """Sweep the CPN-S tolerance and collect one run per achieved ``n``.

    ``build(tol)`` fits and returns a model (``capped_`` set if abandoned),
    ``evaluate(model)`` returns the dict stored for that run.  A geometric
    grid from ``tol_hi`` down (``per_decade`` points per decade) is walked
    until a run is capped (``n > n_max``); gaps in the achieved dimensions are
    then bisected in ``log(tol)`` until every ``n`` between the extremes is
    hit or adjacent tolerances are within ``min_ratio``.  Returns
    ``{"runs": [...], "by_n": {n: run}}`` where ``by_n`` keeps, per ``n``, the
    run with the smallest (tightest) tolerance.  ``prior_runs`` (records of
    an interrupted sweep) are reused, and ``checkpoint(runs_list)`` is called
    after every new fit so the sweep can be resumed.
    """
    runs: dict[float, dict] = {float(r["tol"]): r for r in (prior_runs or [])}

    def run(tol: float) -> dict:
        tol = float(tol)
        if tol in runs:
            return runs[tol]
        model = build(tol)
        if model.capped_:
            rec = dict(tol=tol, n=int(model.n_), N=int(model.N_), capped=True,
                       fit_secs=round(float(model.fit_secs_), 1))
        else:
            rec = dict(evaluate(model), **model.summary())
        runs[tol] = rec
        if checkpoint is not None:
            checkpoint([runs[t] for t in sorted(runs, reverse=True)])
        log(f"    [cpn] tol={tol:.3e}  N={rec['N']:3d}  n={rec['n']:2d}"
            + ("  (capped)" if rec["capped"] else
               f"  train={rec['train']:.2e} t_in={rec['test_interp']:.2e} "
               f"t_ex={rec['test_extrap']:.2e} act={rec['n_active_nonlinear']}")
            + f"  ({rec['fit_secs']}s)")
        return rec

    grid = []
    t = tol_hi
    while t >= tol_lo:
        grid.append(t)
        t /= 10.0 ** (1.0 / per_decade)
    for tol in grid:
        rec = run(tol)
        if rec["capped"]:
            break

    # bisect gaps between consecutive achieved dimensions
    changed = True
    while changed:
        changed = False
        tols = sorted(runs, reverse=True)
        for a, b in zip(tols[:-1], tols[1:]):
            na, nb = runs[a]["n"], runs[b]["n"]
            if nb - na > 1 and a / b > min_ratio and na < n_max:
                run(float(np.sqrt(a * b)))
                changed = True
                break

    by_n: dict[int, dict] = {}
    for tol in sorted(runs, reverse=True):
        rec = runs[tol]
        if rec["capped"]:
            continue
        by_n[rec["n"]] = rec   # later (tighter) tolerances overwrite
    return dict(runs=[runs[t] for t in sorted(runs, reverse=True)],
                by_n={int(k): v for k, v in sorted(by_n.items())})
