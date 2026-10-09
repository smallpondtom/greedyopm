"""Linear advection-diffusion problem: exact spectral data and splits.

The PDE is the periodic linear advection-diffusion equation

    u_t + c u_x = nu u_xx ,    x in [0, 1) ,

with a Gaussian pulse as initial condition and a very small nonzero
viscosity ``nu``, making the dynamics advection-dominant: the snapshot
matrix has slowly decaying singular values (Kolmogorov n-width barrier),
while the weak diffusion slowly widens and damps the pulse.

Parametric splits (train / validation / test), varied parameter = pulse width
------------------------------------------------------------------------------
The wave speed is fixed (``C``, one domain traversal over the time window)
and the varied parameter is the **initial pulse width** ``w`` (the squared
width of the Gaussian, ``u0 = exp(-(x - x0)^2 / w) / sqrt(pi w)``).  Since
diffusion widens a pulse by only ``4 nu T`` over the window, widths outside
the training range by more than that are states no training trajectory
contains, and interior widths are genuine interpolation between training
pulses.  Narrower pulses carry more high-frequency content and need more
modes, so **extrapolation targets a narrower pulse** than any training one.

* ``train``        -- ``N_PARAMS_TRAIN`` widths log-spaced on ``[W_MIN, W_MAX]``.
* ``val_interp``   -- interpolated held-out widths (interior geometric
  midpoints).
* ``val_extrap``   -- a held-out width just below ``W_MIN`` (a *nearer*
  extrapolation than the extrapolation test), so the budget can be tuned to
  stay stable outside the training range.
* ``test_interp``  -- one random interior width (interpolation test).
* ``test_extrap``  -- one width a set factor below ``W_MIN`` (extrapolation
  test, narrower than any training pulse).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# ---- problem configuration --------------------------------------------------

NX = 1024           # spatial grid points (state dimension d)
NU = 1e-4           # viscosity: small but nonzero -> advection-dominant
X0 = 0.2            # initial pulse center
C = 1.0             # wave speed (fixed): one domain traversal per window

T_WINDOW = 1.0      # time window: one full domain traversal at unit speed

# --- parameter regime: the initial pulse width w (squared width) -----------
W_MIN = 1.0e-3      # narrowest training pulse
W_MAX = 4.0e-3      # widest training pulse

N_PARAMS_TRAIN = 4  # number of training widths
N_SNAP_TRAIN = 120  # snapshots per training width  (=> 480 train columns)

N_SNAP_VAL = 100    # snapshots per validation width
N_SNAP_TEST = 300   # snapshots per test width

#: Extrapolation test width as a factor of ``W_MIN`` (narrower pulse).
EXTRAP_FACTOR = 0.5
#: The validation extrapolation sits nearer the range (still held out).
VAL_EXTRAP_FACTOR = 0.8

SEED = 42           # RNG seed selecting the random interpolation test width

SPLITS = ("train", "val_interp", "val_extrap", "test_interp", "test_extrap")


def solve_advection_diffusion(
    pulse_w: float,
    t: np.ndarray,
    *,
    nx: int = NX,
    nu: float = NU,
    x0: float = X0,
    c: float = C,
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


def training_widths() -> np.ndarray:
    """The ``N_PARAMS_TRAIN`` training widths, log-spaced over the regime."""
    return np.geomspace(W_MIN, W_MAX, N_PARAMS_TRAIN)


def val_interp_widths() -> np.ndarray:
    """Interpolated validation widths (interior geometric midpoints)."""
    w_tr = training_widths()
    mids = np.sqrt(w_tr[:-1] * w_tr[1:])
    idx = np.unique(np.linspace(0, mids.shape[0] - 1, 2).round().astype(int))
    return mids[idx]


def test_extrap_width() -> float:
    """The extrapolation test width, a fixed factor below ``W_MIN``."""
    return float(W_MIN * EXTRAP_FACTOR)


def val_extrap_width() -> float:
    """A held-out extrapolation width nearer the range than the test."""
    return float(W_MIN * VAL_EXTRAP_FACTOR)


def test_interp_width(seed: int = SEED, *, tol: float = 0.03) -> float:
    """A single random interior width (interpolation test), log-uniform."""
    nodes = np.log(np.concatenate([training_widths(), val_interp_widths()]))
    rng = np.random.default_rng(seed)
    lo, hi = np.log(W_MIN), np.log(W_MAX)
    span = hi - lo
    while True:
        lw = float(rng.uniform(lo + 0.05 * span, hi - 0.05 * span))
        if np.min(np.abs(nodes - lw)) > tol * span:
            return float(np.exp(lw))


def _stack(widths: np.ndarray, n_snap: int) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate the snapshots of several widths into one matrix."""
    t = np.linspace(0.0, T_WINDOW, n_snap, endpoint=False)
    blocks, x = [], None
    for w in np.atleast_1d(widths):
        S, x = solve_advection_diffusion(float(w), t)
        blocks.append(S)
    return np.concatenate(blocks, axis=1), (t, x)


def make_dataset() -> Dict[str, dict]:
    """Generate the train / validation / test splits (interp and extrap)."""
    w_train = training_widths()
    w_vi = val_interp_widths()
    w_ve = val_extrap_width()
    w_ti = test_interp_width()
    w_te = test_extrap_width()

    S_train, (t_tr, x) = _stack(w_train, N_SNAP_TRAIN)
    S_vi, (t_vi, _) = _stack(w_vi, N_SNAP_VAL)
    S_ve, (t_ve, _) = _stack([w_ve], N_SNAP_VAL)
    S_ti, (t_ti, _) = _stack([w_ti], N_SNAP_TEST)
    S_te, (t_te, _) = _stack([w_te], N_SNAP_TEST)

    return {
        "train": dict(S=S_train, x=x, t=t_tr,
                      widths=list(map(float, w_train)), n_per=N_SNAP_TRAIN),
        "val_interp": dict(S=S_vi, x=x, t=t_vi,
                           widths=list(map(float, w_vi)), n_per=N_SNAP_VAL),
        "val_extrap": dict(S=S_ve, x=x, t=t_ve,
                           widths=[float(w_ve)], n_per=N_SNAP_VAL,
                           w=float(w_ve)),
        "test_interp": dict(S=S_ti, x=x, t=t_ti,
                            widths=[float(w_ti)], n_per=N_SNAP_TEST,
                            w=float(w_ti)),
        "test_extrap": dict(S=S_te, x=x, t=t_te,
                            widths=[float(w_te)], n_per=N_SNAP_TEST,
                            w=float(w_te)),
    }
