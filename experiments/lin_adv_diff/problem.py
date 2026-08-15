"""Linear advection-diffusion problem: exact spectral data and splits.

The PDE is the periodic linear advection-diffusion equation

    u_t + c u_x = nu u_xx ,    x in [0, 1) ,

with a narrow Gaussian pulse as initial condition and a very small nonzero
viscosity ``nu``, making the dynamics advection-dominant: the snapshot
matrix has slowly decaying singular values (Kolmogorov n-width barrier),
while the weak diffusion slowly widens and damps the pulse so that later
times and other wave speeds are genuinely out-of-distribution yet
reconstructible.

The solution is computed exactly in Fourier space:
``u_hat_k(t) = u_hat_k(0) exp(-(2 pi i k c + nu (2 pi k)^2) t)``.

Data splits
-----------
* ``train``          -- wave speed ``C_TRAIN``, one domain traversal
  ``t in [0, T_TRAIN]``.
* ``test_param``     -- unseen wave speed ``C_TEST`` (the only varied
  parameter), same time window as training.
* ``test_forecast``  -- wave speed ``C_TRAIN``, contiguous future window
  ``t in (T_TRAIN, T_FINAL]``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# ---- problem configuration --------------------------------------------------

NX = 1024          # spatial grid points (state dimension d)
NU = 1e-4          # viscosity: small but nonzero -> advection-dominant
C_TRAIN = 1.0      # training wave speed
C_TEST = 1.2       # unseen test wave speed (the single varied parameter)
X0 = 0.2           # initial pulse center
PULSE_W = 1e-3     # squared pulse width of the Gaussian initial condition

T_TRAIN = 1.0      # training window: one full domain traversal at C_TRAIN
T_FINAL = 2.0      # forecast horizon
N_TRAIN = 1200     # training snapshots  (>= feature count of p=4 at r=10)
N_TEST = 600       # snapshots in each test window


def solve_advection_diffusion(
    c: float,
    t: np.ndarray,
    *,
    nx: int = NX,
    nu: float = NU,
    x0: float = X0,
    pulse_w: float = PULSE_W,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact spectral solution snapshots for the given time samples.

    Arguments
    ---------
    c : float
        Wave speed.
    t : (n,) array
        Time samples.
    nx, nu, x0, pulse_w :
        Grid size, viscosity, pulse center, squared pulse width.

    Returns
    -------
    S : (nx, n) snapshot matrix.
    x : (nx,) spatial grid.
    """
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    u0 = np.exp(-((x - x0) ** 2) / pulse_w) / np.sqrt(np.pi * pulse_w)

    k = np.fft.rfftfreq(nx, d=1.0 / nx)                 # integer wavenumbers
    u0_hat = np.fft.rfft(u0)
    decay = -(2j * np.pi * k * c + nu * (2.0 * np.pi * k) ** 2)
    S_hat = u0_hat[:, None] * np.exp(decay[:, None] * t[None, :])
    S = np.fft.irfft(S_hat, n=nx, axis=0)
    return np.ascontiguousarray(S), x


def make_dataset() -> Dict[str, dict]:
    """Generate the three splits used by the experiment.

    Returns
    -------
    dict with keys ``train``, ``test_param``, ``test_forecast``; each value
    holds ``S`` (nx, n), ``x`` (nx,), ``t`` (n,), and the wave speed ``c``.
    """
    t_train = np.linspace(0.0, T_TRAIN, N_TRAIN, endpoint=False)
    t_forecast = np.linspace(T_TRAIN, T_FINAL, N_TEST, endpoint=False)
    t_param = np.linspace(0.0, T_TRAIN, N_TEST, endpoint=False)

    S_train, x = solve_advection_diffusion(C_TRAIN, t_train)
    S_forecast, _ = solve_advection_diffusion(C_TRAIN, t_forecast)
    S_param, _ = solve_advection_diffusion(C_TEST, t_param)

    return {
        "train": dict(S=S_train, x=x, t=t_train, c=C_TRAIN),
        "test_param": dict(S=S_param, x=x, t=t_param, c=C_TEST),
        "test_forecast": dict(S=S_forecast, x=x, t=t_forecast, c=C_TRAIN),
    }
