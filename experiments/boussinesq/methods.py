"""Method registry for the 2D Boussinesq rising-thermal comparison.

Six methods are compared:

* ``POD``               -- leading-r linear projection (linear reference).
* ``Greedy-QM``         -- greedy quadratic manifold, full monomial shell.
* ``Greedy-CM``         -- plain greedy manifold, all degree-2 and 3 monomials.
* ``Geelen-QM``         -- leading-r POD + group-sparse quadratic columns.
* ``Greedy-OPM (p=2)``  -- proposed method, quadratic orthonormal features.
* ``Greedy-OPM (p=3)``  -- proposed method, degrees {2, 3}.

The dilation ``delta`` is fixed; the Lipschitz budget ``L`` is the only
tuned hyperparameter and is **not** stored here.  It is selected -- once for
``p=2`` and once for ``p=3`` -- by the grid-adaptive sweep in
:mod:`budget_select`, which balances the interpolation and extrapolation
validation splits, and passed to :func:`build_model` explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                        # baselines, problem
sys.path.insert(0, str(_HERE.parent.parent / "src"))  # greedyopm
sys.path.insert(0, str(_HERE.parent))               # cpn_baseline

jax.config.update("jax_enable_x64", True)

from baselines import POD, GeelenQM, GreedyCM, GreedyQM  # noqa: E402
from greedyopm import GreedyOPM                       # noqa: E402
from preprocess import PreprocessedModel, Normalizer  # noqa: E402


# ---- shared experiment configuration ----------------------------------------

CFG = dict(
    s_pool=50,              # candidate pool size for the greedy methods
    delta=0.1,              # extrapolation margin for the certificate
    n_grad_samples=150,     # gradient probe subsample in the greedy loop
    n_reorth=4,             # extra Gram-Schmidt passes
    qm_reg=np.sqrt(1e-13),  # Greedy-QM ridge magnitude
    cm_reg=np.sqrt(1e-13),  # Greedy-CM ridge magnitude
    geelen_rel=0.01,        # Geelen-QM group-lasso weight (relative)
    geelen_gamma=1e-13,     # Geelen-QM refit Tikhonov
    geelen_iters=400,       # Geelen-QM FISTA iterations
)

METHOD_NAMES = [
    "POD",
    "Greedy-QM",
    "Greedy-CM",
    "Geelen-QM",
    "Greedy-OPM (p=2)",
    "Greedy-OPM (p=3)",
]

OPM_DEGREES = {
    "Greedy-OPM (p=2)": [2],
    "Greedy-OPM (p=3)": [2, 3],
}

#: CPN-S (Bensalah, Nouy & Soffo 2026) is fitted by a tolerance sweep rather
#: than at a prescribed r (see :mod:`cpn_baseline`), so it is not part of the
#: r-sweep registry ``METHOD_NAMES``; it joins the comparison in the plots.
CPN_NAME = "CPN-S"

#: Display order of the error-vs-r curves.
PLOT_METHODS = [
    "POD",
    "Geelen-QM",
    CPN_NAME,
    "Greedy-QM",
    "Greedy-OPM (p=2)",
    "Greedy-CM",
    "Greedy-OPM (p=3)",
]

#: Display order of the active-feature bars (no POD; each plain greedy
#: manifold sits immediately left of the Greedy-OPM of the same degree).
#: CPN-S is left out: its count (distinct basis functions over 40 composed
#: tail functions, see cpn_baseline) is not the same quantity; it is
#: reported in the report JSON instead.
BAR_METHODS = [
    "Geelen-QM",
    "Greedy-QM",
    "Greedy-OPM (p=2)",
    "Greedy-CM",
    "Greedy-OPM (p=3)",
]

#: Legend text (the internal names above are the JSON keys and stay fixed).
LEGEND_LABELS = {
    "POD": "POD",
    "Geelen-QM": "POD-Encoder QM",
    "Greedy-QM": "Plain Greedy (p=2)",
    "Greedy-CM": "Plain Greedy (p=3)",
    "Greedy-OPM (p=2)": "Greedy-OPM (p=2)",
    "Greedy-OPM (p=3)": "Greedy-OPM (p=3)",
    CPN_NAME: "CPN-S (p=3)",
}

#: CPN-S settings: the released defaults of ``configs/cpn.yaml`` (degree 3,
#: alpha = 1, beta = 0.7071, L = 100, n_min = 1, no internal validation
#: split, mean-squared setting), including their own choice of the
#: background space ``X_N`` from the tolerance; only the reduced dimension
#: is capped at ``n <= R_MAX`` by the tolerance sweep of the driver.
CPN_CFG = dict(
    N=None,                         # their truncation rule: N = N(tol) is the
                                    # smallest PCA truncation whose tail energy
                                    # is <= (beta tol ||S||_F)^2 (Sec. 4.1);
                                    # an integer would fix N instead
    p=3,
    alpha=1.0,
    beta=0.7071,
    L=100.0,
    n_min=1,
    train_val_set=1.0,
    index_set="hyperbolic_cross",   # the released default (prod(nu_k + 1)
                                    # <= p + 1); "total_degree" is the
                                    # full |nu|_1 <= p dictionary
)


def set_cpn_degree(p: int) -> None:
    """Set the CPN-S polynomial degree (per layer) and its legend label."""
    CPN_CFG["p"] = int(p)
    LEGEND_LABELS[CPN_NAME] = f"CPN-S (p={int(p)})"


METHOD_STYLES = {
    "POD":              dict(fmt="-o",  color="black",   lw=3.4, ms=10),
    "Greedy-QM":        dict(fmt="--^", color="#1f77b4", lw=3.0, ms=10),
    "Greedy-CM":        dict(fmt="--s", color="#2ca02c", lw=3.0, ms=10),
    "Geelen-QM":        dict(fmt="--v", color="#ff7f0e", lw=3.0, ms=10),
    "Greedy-OPM (p=2)": dict(fmt="-P",  color="#9467bd", lw=3.0, ms=11),
    "Greedy-OPM (p=3)": dict(fmt="-D",  color="#d62728", lw=3.0, ms=10),
    CPN_NAME:           dict(fmt="-.X", color="#8c564b", lw=3.0, ms=11),
}


def build_model(name: str, S_train, r: int, cfg: dict = CFG, *, L=None):
    """Build and fit a single model on the training snapshots.

    The snapshots (full states, see :mod:`problem`) are centered
    by the training row mean (see :mod:`preprocess`); every method is fit on
    the centered data, and the returned wrapper inverts the preprocessing at
    reconstruction so all errors are exact full-state errors in physical
    units.

    Arguments
    ---------
    name : str
        One of ``METHOD_NAMES``.
    S_train : (d, n)
        Raw training snapshot matrix.
    r : int
        Reduced dimension.
    cfg : dict
        Shared configuration (see ``CFG``).
    L : float
        Lipschitz budget for the Greedy-OPM variants.  There is no default
        and no hard-coded value: the driver selects it on the validation
        splits (see :mod:`budget_select`) and passes it in explicitly.

    Returns
    -------
    A fitted :class:`preprocess.PreprocessedModel` exposing
    ``reconstruct`` / ``reconstruction_error`` in the original units.
    """
    normalizer = Normalizer().fit(S_train)
    N_train = normalizer.transform(S_train)

    if name == "POD":
        model = POD(r).fit(N_train)
    elif name == "Greedy-QM":
        model = GreedyQM(r, cfg["s_pool"], reg=cfg["qm_reg"]).fit(N_train)
    elif name == "Greedy-CM":
        model = GreedyCM(r, cfg["s_pool"], reg=cfg["cm_reg"]).fit(N_train)
    elif name == "Geelen-QM":
        model = GeelenQM(
            r, gamma_sp_rel=cfg["geelen_rel"], gamma=cfg["geelen_gamma"],
            n_iter=cfg["geelen_iters"],
        ).fit(N_train)
    elif name in OPM_DEGREES:
        if L is None:
            raise ValueError(
                f"{name} requires an explicit Lipschitz budget L "
                "(selected on the validation splits by budget_select)."
            )
        model = GreedyOPM(
            r, cfg["s_pool"], OPM_DEGREES[name],
            L=float(L),
            delta=cfg["delta"],
            n_reorth=cfg["n_reorth"],
            n_grad_samples=cfg["n_grad_samples"],
        ).fit(N_train)
    else:
        raise ValueError(f"Unknown method {name!r}")
    return PreprocessedModel(model, normalizer)


def build_cpn(S_train, tol: float, *, n_max: int | None = None,
              cfg: dict = CPN_CFG):
    """Fit CPN-S (released code) at target tolerance ``tol``.

    Same preprocessing as :func:`build_model`; the manifold dimension is an
    outcome of ``tol`` (``model.n_``), the run is abandoned (``capped_``) if
    more than ``n_max`` modes get promoted.
    """
    from cpn_baseline import CPNSparse

    normalizer = Normalizer().fit(S_train)
    N_train = np.asarray(normalizer.transform(S_train))
    model = CPNSparse(tol, n_max=n_max, **cfg).fit(N_train)
    return PreprocessedModel(model, normalizer)


def active_by_degree(model) -> dict | None:
    """``{degree: count}`` of active nonlinear features, when tracked."""
    if hasattr(model, "active_degree_counts"):
        return {str(k): int(v) for k, v in model.active_degree_counts.items()}
    return None


def n_active(model) -> int:
    """Number of active nonlinear features of a fitted model."""
    return int(getattr(model, "n_active_nonlinear", 0))
