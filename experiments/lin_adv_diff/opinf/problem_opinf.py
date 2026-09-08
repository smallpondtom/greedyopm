"""Parametric trajectory data for the OpInf-on-manifold experiment.

Two configurations of the periodic linear advection-diffusion problem are
provided; both share the solver of :mod:`problem` one level up.

* ``"gentle"`` (default) -- the regime in which the reduced dynamics are the
  question being asked.  Softer physics (``nu = 2e-3``, pulse width ``8e-3``),
  a narrower speed range ``[1, 5]``, and a *denser* parameter sampling (nine
  training speeds) so the per-speed operators and their spline interpolation
  are well supported.  At moderate ``r`` the Greedy-OPM manifold has a clear
  representational advantage over POD at budgets the dynamics can tolerate, so
  the reduced-model comparison measures something that exists.

* ``"manifold"`` -- the settings of the reconstruction (manifold-learning)
  experiment, verbatim: ``nu = 1e-4``, near-delta pulse, speeds ``{1,4,7,10}``.
  This is a deliberately brutal stress case: at ``r <= 10`` every ROM-stable
  manifold carries ~50% representation error, so no latent dynamics can beat
  its decoder floor -- the configuration demonstrates *stability* (nothing but
  budgeted GMG-OPM survives), not accuracy.

In both regimes every trajectory starts from the same Gaussian pulse (the
initial condition is parameter-independent), the manifold is fit on a
concatenated snapshot matrix over the training speeds, and Operator Inference
uses densely sampled trajectories so the FBCT4 velocity estimates are resolved
at the fastest speed.  The held-out roles: two interior *validation* speeds
(ROM budget selection) and one interior *interpolation test* speed.
Extrapolated speeds are not evaluated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # ../ (problem, methods, ...)

import problem as prob  # noqa: E402

REGIMES = ("gentle", "manifold")
DEFAULT_REGIME = "gentle"

#: gentle-regime physics and parameter sampling
GENTLE = dict(
    nu=2.0e-3, pulse_w=8.0e-3,
    t_window=0.35, n_time=1000,
    train_speeds=[float(c) for c in np.linspace(1.0, 5.0, 9)],
    val_speeds=[2.25, 2.75, 4.25],
    c_interp=3.3,
    fit_stride=10,  # 100 fit columns per speed -> 900 total
)


def _trajectory(c: float, t: np.ndarray, nu: float, pw: float) -> np.ndarray:
    S, _ = prob.solve_advection_diffusion(float(c), t, nu=nu, pulse_w=pw)
    return np.ascontiguousarray(S)


def exact_velocity(S: np.ndarray, c: float, nu: float) -> np.ndarray:
    """Exact time derivative ``u_t = -c u_x + nu u_xx`` by FFT.

    The snapshots are the exact spectral solution on the uniform periodic
    grid, so spectral differentiation of each column is the exact velocity --
    no finite-difference truncation error in the OpInf targets.
    """
    S = np.asarray(S, dtype=float)
    d = S.shape[0]
    k = np.fft.rfftfreq(d, d=1.0 / d)                   # integer wavenumbers
    mult = -(2j * np.pi * k * float(c) + float(nu) * (2.0 * np.pi * k) ** 2)
    return np.ascontiguousarray(
        np.fft.irfft(mult[:, None] * np.fft.rfft(S, axis=0), n=d, axis=0))


def _gentle_dataset() -> Dict:
    g = GENTLE
    t = np.linspace(0.0, g["t_window"], g["n_time"], endpoint=False)
    dt = float(t[1] - t[0])
    x = np.linspace(0.0, 1.0, prob.NX, endpoint=False)
    nu, pw = g["nu"], g["pulse_w"]

    blocks: List[np.ndarray] = [_trajectory(c, t, nu, pw)
                                for c in g["train_speeds"]]
    dblocks = [exact_velocity(blk, c, nu)
               for blk, c in zip(blocks, g["train_speeds"])]
    S_fit = np.concatenate([blk[:, ::g["fit_stride"]] for blk in blocks],
                           axis=1)
    return dict(
        regime="gentle", nu=nu, pulse_w=pw, t=t, dt=dt, x=x,
        n_time=g["n_time"], t_window=g["t_window"],
        train=dict(speeds=list(g["train_speeds"]), blocks=blocks,
                   dblocks=dblocks, S_fit=S_fit,
                   n_fit_per=blocks[0][:, ::g["fit_stride"]].shape[1]),
        val=dict(speeds=list(g["val_speeds"]),
                 blocks=[_trajectory(c, t, nu, pw) for c in g["val_speeds"]]),
        interp=dict(c=float(g["c_interp"]),
                    S=_trajectory(g["c_interp"], t, nu, pw)),
    )


def _manifold_dataset() -> Dict:
    """The reconstruction-experiment settings, verbatim (see module docstring)."""
    n_time = 2000
    t = np.linspace(0.0, prob.T_WINDOW, n_time, endpoint=False)
    dt = float(t[1] - t[0])
    x = np.linspace(0.0, 1.0, prob.NX, endpoint=False)
    nu, pw = prob.NU, prob.PULSE_W

    speeds = [float(c) for c in prob.training_speeds()]
    blocks = [_trajectory(c, t, nu, pw) for c in speeds]
    dblocks = [exact_velocity(blk, c, nu) for blk, c in zip(blocks, speeds)]
    t_fit = np.linspace(0.0, prob.T_WINDOW, prob.N_SNAP_TRAIN, endpoint=False)
    S_fit = np.concatenate([_trajectory(c, t_fit, nu, pw) for c in speeds],
                           axis=1)
    return dict(
        regime="manifold", nu=nu, pulse_w=pw, t=t, dt=dt, x=x,
        n_time=n_time, t_window=prob.T_WINDOW,
        train=dict(speeds=speeds, blocks=blocks, dblocks=dblocks, S_fit=S_fit,
                   n_fit_per=prob.N_SNAP_TRAIN),
        val=dict(speeds=[float(c) for c in prob.val_interp_speeds()],
                 blocks=[_trajectory(float(c), t, nu, pw)
                         for c in prob.val_interp_speeds()]),
        interp=dict(c=float(prob.test_interp_speed()),
                    S=_trajectory(prob.test_interp_speed(), t, nu, pw)),
    )


def make_opinf_dataset(regime: str = DEFAULT_REGIME) -> Dict:
    if regime == "gentle":
        return _gentle_dataset()
    if regime == "manifold":
        return _manifold_dataset()
    raise ValueError(f"Unknown regime {regime!r}; choose one of {REGIMES}.")


def summary(data: Dict) -> str:
    d = data["train"]["blocks"][0].shape[0]
    P = len(data["train"]["speeds"])
    return (
        f"regime={data['regime']} nu={data['nu']:g} pw={data['pulse_w']:g}  "
        f"d={d} P_train={P} speeds={np.round(data['train']['speeds'], 2).tolist()}  "
        f"N_time={data['n_time']} dt={data['dt']:.2e} T={data['t_window']}  "
        f"fit_cols={data['train']['S_fit'].shape[1]}  "
        f"val={data['val']['speeds']} interp c={data['interp']['c']:.4g}"
    )
