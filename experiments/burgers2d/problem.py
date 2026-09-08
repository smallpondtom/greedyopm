"""2D viscous Burgers problem: solver data and the parametric splits.

The system is the 2D viscous Burgers equation on the periodic square
``[0, 2 pi)^2``

    u_t + u u_x + v u_y = nu (u_xx + u_yy) ,
    v_t + u v_x + v v_y = nu (v_xx + v_yy) ,

integrated with the Fourier pseudo-spectral / RK4 solver of
:mod:`experiments.data.burgers2d` from a two-scale vortical initial
condition on a uniform mean flow ``(c, c)``.  The modelled state stacks both
velocity components, ``d = 2 N^2``.

Why this problem
----------------
The nonlinearity steepens the initial harmonics into thin internal layers,
the viscosity ``nu`` sets their thickness, and the mean flow sweeps them
diagonally across the box -- transport of thin moving structures gives a
slowly decaying Kolmogorov n-width (a linear subspace struggles).  Because a
constant mean flow only translates the solution (Galilean invariance), the
Reynolds number ``Re = amp * 2 pi / nu`` -- not the advection speed -- is the
parameter that changes the *shape* of the solution, and it is what the
parametric test varies.  Large ``Re`` is the near-inviscid regime; the range
here stays fully resolved on the ``128^2`` grid (spectral tail < 1e-6).

Parametric splits (train / validation / test)
---------------------------------------------
The training Reynolds numbers sample a (deliberately wide) range
``[RE_MIN, RE_MAX]``.  There are **two** test scenarios, mirrored by the
validation set so the Lipschitz budget can be tuned for both:

* ``train``        -- ``N_PARAMS_TRAIN`` numbers evenly on ``[RE_MIN, RE_MAX]``.
* ``val_interp``   -- interpolated held-out Reynolds numbers (midpoints).
* ``val_extrap``   -- a held-out number just beyond ``RE_MAX`` (a *nearer*
  extrapolation than the extrapolation test).
* ``test_interp``  -- one random interior number (interpolation test).
* ``test_extrap``  -- one number a set fraction beyond ``RE_MAX``
  (extrapolation test).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data"

# ---- problem configuration --------------------------------------------------

N = 128              # grid points per direction (state dimension d = 2 N^2)
DT = 1.0e-3          # RK4 time step
T_END = 2.0 * np.pi  # final time: one full domain traversal at C_MEAN
C_MEAN = 1.0         # uniform mean flow in both directions
AMP = 1.0            # amplitude of the vortical perturbation

# --- parameter regime: the Reynolds number Re = amp * 2 pi / nu (widened) ---
RE_MIN = 140.0       # smallest training Reynolds number (nu ~ 0.0449)
RE_MAX = 340.0       # largest  training Reynolds number (nu ~ 0.0185)

N_PARAMS_TRAIN = 4       # number of training Reynolds numbers
STORE_EVERY_TRAIN = 74   # snapshot stride  ->  ~85 snapshots per Re (sparse)
STORE_EVERY_VAL = 80     # coarser stride for the validation sets (~79 each)
STORE_EVERY_TEST = 48    # stride for the test Reynolds numbers (~131 each)

#: Extrapolation distance of the test, as a fraction of the training span
#: (far enough that a loose budget over-extrapolates -- the budget bites).
EXTRAP_FRAC = 1.2
#: The validation extrapolation sits nearer the range (held out).
VAL_EXTRAP_FRAC = 0.6
SEED = 0                 # RNG seed selecting the random interpolation test Re

SPLITS = ("train", "val_interp", "val_extrap", "test_interp", "test_extrap")


def reynolds(nu: float) -> float:
    """``Re = amp * L / nu`` with ``L = 2 pi`` the domain size."""
    return AMP * 2.0 * np.pi / float(nu)


def viscosity(Re: float) -> float:
    """Inverse of :func:`reynolds`: ``nu = amp * 2 pi / Re``."""
    return AMP * 2.0 * np.pi / float(Re)


def solve_burgers(nu: float, store_every: int) -> Tuple[np.ndarray, dict]:
    """Velocity snapshots ``[u; v]`` for one viscosity (disk-cached)."""
    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA / f"burgers2d_N{N}_nu{nu:.6g}_se{store_every}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["S"], dict(X=z["X"], Y=z["Y"], tspan=z["tspan"],
                            field_shape=(N, N), nu=float(nu),
                            Re=reynolds(nu))

    # import sys
    # sys.path.insert(0, str(_HERE.parent / "data"))
    from burgers2d import generate_burgers2d   # noqa: E402

    S, info = generate_burgers2d(
        N=N, nu=float(nu), dt=DT, t_end=T_END, store_every=int(store_every),
        c=C_MEAN, amp=AMP,
    )
    np.savez(cache, S=S, X=info["X"], Y=info["Y"], tspan=info["tspan"])
    return S, dict(X=info["X"], Y=info["Y"], tspan=info["tspan"],
                   field_shape=(N, N), nu=float(nu), Re=info["Re"])


# ---- parameter sampling -----------------------------------------------------


def training_reynolds() -> np.ndarray:
    """The ``N_PARAMS_TRAIN`` training Reynolds numbers spanning the regime."""
    return np.linspace(RE_MIN, RE_MAX, N_PARAMS_TRAIN)


def val_interp_reynolds() -> np.ndarray:
    """Interpolated validation Reynolds numbers (interior midpoints)."""
    re_tr = training_reynolds()
    mids = 0.5 * (re_tr[:-1] + re_tr[1:])
    idx = np.unique(np.linspace(0, mids.shape[0] - 1, 2).round().astype(int))
    return mids[idx]


def test_extrap_reynolds() -> float:
    """The extrapolation test Reynolds number, beyond ``RE_MAX``."""
    return float(RE_MAX + EXTRAP_FRAC * (RE_MAX - RE_MIN))


def val_extrap_reynolds() -> float:
    """A held-out extrapolation Reynolds number nearer the range."""
    return float(RE_MAX + VAL_EXTRAP_FRAC * (RE_MAX - RE_MIN))


def test_interp_reynolds(seed: int = SEED, *, tol_frac: float = 0.03) -> float:
    """A single random interior Reynolds number (interpolation test)."""
    nodes = np.concatenate([training_reynolds(), val_interp_reynolds()])
    rng = np.random.default_rng(seed)
    span = RE_MAX - RE_MIN
    while True:
        Re = float(rng.uniform(RE_MIN + 0.05 * span, RE_MAX - 0.05 * span))
        if np.min(np.abs(nodes - Re)) > tol_frac * span:
            return Re


def _stack(reynolds_numbers, store_every: int) -> Tuple[np.ndarray, dict]:
    """Concatenate the snapshots of several Reynolds numbers into one matrix."""
    blocks, info = [], None
    for Re in np.atleast_1d(reynolds_numbers):
        S, info = solve_burgers(viscosity(float(Re)), store_every)
        blocks.append(S)
    return np.concatenate(blocks, axis=1), info


def make_dataset() -> Dict[str, dict]:
    """Generate the train / validation / test splits (interp and extrap)."""
    re_train = training_reynolds()
    re_vi = val_interp_reynolds()
    re_ve = val_extrap_reynolds()
    re_ti = test_interp_reynolds()
    re_te = test_extrap_reynolds()

    S_train, info = _stack(re_train, STORE_EVERY_TRAIN)
    S_vi, _ = _stack(re_vi, STORE_EVERY_VAL)
    S_ve, _ = _stack([re_ve], STORE_EVERY_VAL)
    S_ti, it = _stack([re_ti], STORE_EVERY_TEST)
    S_te, te = _stack([re_te], STORE_EVERY_TEST)

    def pack(S, Res, inf, extra=None):
        d = dict(S=S, X=inf["X"], Y=inf["Y"], t=np.asarray(inf["tspan"]),
                 Re=list(map(float, np.atleast_1d(Res))), field_shape=(N, N))
        if extra is not None:
            d.update(extra)
        return d

    return {
        "train": pack(S_train, re_train, info),
        "val_interp": pack(S_vi, re_vi, info),
        "val_extrap": pack(S_ve, [re_ve], info,
                           dict(nu=float(viscosity(re_ve)))),
        "test_interp": pack(S_ti, [re_ti], it,
                            dict(nu=float(viscosity(re_ti)))),
        "test_extrap": pack(S_te, [re_te], te,
                            dict(nu=float(viscosity(re_te)))),
    }


def speed(col: np.ndarray) -> np.ndarray:
    """Velocity magnitude ``sqrt(u^2 + v^2)`` of one stacked snapshot."""
    col = np.asarray(col)
    NN = N * N
    return np.sqrt(col[:NN] ** 2 + col[NN:] ** 2).reshape(N, N)
