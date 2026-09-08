"""Linear advection-diffusion problem: exact spectral data and splits.

The PDE is the periodic linear advection-diffusion equation

    u_t + c u_x = nu u_xx ,    x in [0, 1) ,

with a narrow Gaussian pulse as initial condition and a very small nonzero
viscosity ``nu``, making the dynamics advection-dominant: the snapshot
matrix has slowly decaying singular values (Kolmogorov n-width barrier),
while the weak diffusion slowly widens and damps the pulse.

Parametric splits (train / validation / test), varied parameter = wave speed
-----------------------------------------------------------------------------
The training wave speeds sample a (deliberately wide) range
``[C_MIN, C_MAX]``.  There are then **two** test scenarios, and the
validation set mirrors both so the Lipschitz budget can be tuned for each:

* ``train``        -- ``N_PARAMS_TRAIN`` speeds evenly on ``[C_MIN, C_MAX]``.
* ``val_interp``   -- interpolated held-out speeds (interior midpoints).
* ``val_extrap``   -- a held-out speed just beyond ``C_MAX`` (a *nearer*
  extrapolation than the extrapolation test), so the budget can be tuned to
  stay stable outside the training range.
* ``test_interp``  -- one random interior speed (interpolation test).
* ``test_extrap``  -- one speed a set fraction beyond ``C_MAX``
  (extrapolation test).

Widening the range and adding the extrapolation test is what exposes the
Lipschitz budget's value: an unregularised polynomial manifold extrapolates
wildly past the training range, while the budgeted decoder has a certified
Lipschitz constant and stays bounded.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# ---- problem configuration --------------------------------------------------

NX = 1024           # spatial grid points (state dimension d)
NU = 1e-4           # viscosity: small but nonzero -> advection-dominant
X0 = 0.2            # initial pulse center
PULSE_W = 1e-3      # squared pulse width of the Gaussian initial condition

T_WINDOW = 1.0      # time window: one full domain traversal at unit speed

# --- parameter regime: the wave speed c (widened) ---------------------------
C_MIN = 1.0         # smallest training wave speed
C_MAX = 10.0        # largest training wave speed  (wide range -> harder)

N_PARAMS_TRAIN = 4  # number of training wave speeds
N_SNAP_TRAIN = 120  # snapshots per training speed  (=> 480 train columns)

N_SNAP_VAL = 100    # snapshots per validation speed
N_SNAP_TEST = 300   # snapshots per test speed

#: Extrapolation distance of the test, as a fraction of the training span.
EXTRAP_FRAC = 2.0
#: The validation extrapolation sits at half that distance (a nearer, still
#: held-out, extrapolation used only to tune the budget).
VAL_EXTRAP_FRAC = 0.4

SEED = 42           # RNG seed selecting the random interpolation test speed

SPLITS = ("train", "val_interp", "val_extrap", "test_interp", "test_extrap")


def solve_advection_diffusion(
    c: float,
    t: np.ndarray,
    *,
    nx: int = NX,
    nu: float = NU,
    x0: float = X0,
    pulse_w: float = PULSE_W,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact spectral solution snapshots for the given time samples."""
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    u0 = np.exp(-((x - x0) ** 2) / pulse_w) / np.sqrt(np.pi * pulse_w)

    k = np.fft.rfftfreq(nx, d=1.0 / nx)                 # integer wavenumbers
    u0_hat = np.fft.rfft(u0)
    decay = -(2j * np.pi * k * c + nu * (2.0 * np.pi * k) ** 2)
    S_hat = u0_hat[:, None] * np.exp(decay[:, None] * t[None, :])
    S = np.fft.irfft(S_hat, n=nx, axis=0)
    return np.ascontiguousarray(S), x


# ---- parameter sampling -----------------------------------------------------


def training_speeds() -> np.ndarray:
    """The ``N_PARAMS_TRAIN`` training wave speeds spanning the regime."""
    return np.linspace(C_MIN, C_MAX, N_PARAMS_TRAIN)


def val_interp_speeds() -> np.ndarray:
    """Interpolated validation speeds (interior midpoints of the grid)."""
    c_tr = training_speeds()
    mids = 0.5 * (c_tr[:-1] + c_tr[1:])
    idx = np.unique(np.linspace(0, mids.shape[0] - 1, 2).round().astype(int))
    return mids[idx]


def test_extrap_speed() -> float:
    """The extrapolation test speed, a fixed fraction beyond ``C_MAX``."""
    return float(C_MAX + EXTRAP_FRAC * (C_MAX - C_MIN))


def val_extrap_speed() -> float:
    """A held-out extrapolation speed nearer the range than the test."""
    return float(C_MAX + VAL_EXTRAP_FRAC * (C_MAX - C_MIN))


def test_interp_speed(seed: int = SEED, *, tol: float = 0.03) -> float:
    """A single random interior speed (interpolation test)."""
    nodes = np.concatenate([training_speeds(), val_interp_speeds()])
    rng = np.random.default_rng(seed)
    span = C_MAX - C_MIN
    while True:
        c = float(rng.uniform(C_MIN + 0.05 * span, C_MAX - 0.05 * span))
        if np.min(np.abs(nodes - c)) > tol * span:
            return c


def _stack(speeds: np.ndarray, n_snap: int) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate the snapshots of several wave speeds into one matrix."""
    t = np.linspace(0.0, T_WINDOW, n_snap, endpoint=False)
    blocks, x = [], None
    for c in np.atleast_1d(speeds):
        S, x = solve_advection_diffusion(float(c), t)
        blocks.append(S)
    return np.concatenate(blocks, axis=1), (t, x)


def make_dataset() -> Dict[str, dict]:
    """Generate the train / validation / test splits (interp and extrap)."""
    c_train = training_speeds()
    c_vi = val_interp_speeds()
    c_ve = val_extrap_speed()
    c_ti = test_interp_speed()
    c_te = test_extrap_speed()

    S_train, (t_tr, x) = _stack(c_train, N_SNAP_TRAIN)
    S_vi, (t_vi, _) = _stack(c_vi, N_SNAP_VAL)
    S_ve, (t_ve, _) = _stack([c_ve], N_SNAP_VAL)
    S_ti, (t_ti, _) = _stack([c_ti], N_SNAP_TEST)
    S_te, (t_te, _) = _stack([c_te], N_SNAP_TEST)

    return {
        "train": dict(S=S_train, x=x, t=t_tr,
                      speeds=list(map(float, c_train)), n_per=N_SNAP_TRAIN),
        "val_interp": dict(S=S_vi, x=x, t=t_vi,
                           speeds=list(map(float, c_vi)), n_per=N_SNAP_VAL),
        "val_extrap": dict(S=S_ve, x=x, t=t_ve,
                           speeds=[float(c_ve)], n_per=N_SNAP_VAL,
                           c=float(c_ve)),
        "test_interp": dict(S=S_ti, x=x, t=t_ti,
                            speeds=[float(c_ti)], n_per=N_SNAP_TEST,
                            c=float(c_ti)),
        "test_extrap": dict(S=S_te, x=x, t=t_te,
                            speeds=[float(c_te)], n_per=N_SNAP_TEST,
                            c=float(c_te)),
    }
