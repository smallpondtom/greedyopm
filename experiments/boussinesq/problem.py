"""2D Boussinesq rising thermal: solver data and the parametric splits.

The system is the incompressible Boussinesq equations on the periodic box
``[0, Lx) x [0, Ly)`` in vorticity-streamfunction form

    w_t  + u w_x  + v w_y  = nu    (w_xx  + w_yy)  + theta_x ,
    th_t + u th_x + v th_y = kappa (th_xx + th_yy) ,

integrated with the Fourier pseudo-spectral / IF-RK4 solver of
:mod:`boussinesq` from a quiescent fluid with a hot Gaussian blob.  The
modelled state stacks the two velocity components and the temperature,
``d = 3 Nx Ny``.

Why this problem
----------------
The blob rises many radii and rolls up into a mushroom cap (a buoyant vortex
pair) whose lobes wind up thin sheets of temperature and vorticity; as the
Reynolds number grows the sheets get thinner and secondary rolls appear on
the cap and stem.  Transport over a long distance gives a slowly decaying
Kolmogorov n-width, and ``Re`` changes the *shape* of the solution (layer
thickness, roll-up detail), so extrapolating in ``Re`` means predicting
finer structure than the training set contains.  Everything is
non-dimensionalized by the blob radius and the buoyancy velocity, so
``Re = U R / nu`` (with ``Pr = 1``, ``Ra = Re^2``) is the only parameter.

Parametric splits (train / validation / test)
---------------------------------------------
Same protocol as ``lin_adv_diff`` and ``burgers2d``:

* ``train``        -- ``N_PARAMS_TRAIN`` numbers evenly on ``[RE_MIN, RE_MAX]``.
* ``val_interp``   -- interpolated held-out Reynolds numbers (midpoints).
* ``val_extrap``   -- a held-out number just beyond ``RE_MAX``.
* ``test_interp``  -- one random interior number (interpolation test).
* ``test_extrap``  -- one number a set fraction beyond ``RE_MAX``.

The whole range, including the extrapolation test, is resolved on the
``NX x NY`` grid; ``validate_solver.py`` checks this against a grid twice
as fine at the largest Reynolds number used.

Data layout
-----------
``make_dataset`` loads the cached snapshot files of every split into memory
in the full state dimension ``d = 3 Nx Ny = 344064`` (float64, about 2.9 GB
for all five splits); all fits and errors are computed directly on these
states, exactly as in ``lin_adv_diff`` and ``burgers2d``.  The files are
written by ``generate_data.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data"

# ---- problem configuration --------------------------------------------------

NX, NY = 256, 448        # grid
D = 3 * NX * NY          # state dimension d = 344064
LX, LY = 8.0, 14.0       # box in blob radii (dx = dy = 1/32)
Y0 = 3.0                 # initial height of the blob (centred in x)
R = 1.0                  # blob radius (the length scale)
PR = 1.0                 # Prandtl number
DT = 5.0e-3              # IF-RK4 time step (advective CFL ~ 0.25)
T_END = 10.0             # final time (cap rises ~7 radii)

# --- parameter regime: the Reynolds number Re = U R / nu ---------------------
RE_MIN = 250.0           # smallest training Reynolds number
RE_MAX = 750.0           # largest  training Reynolds number

N_PARAMS_TRAIN = 4       # number of training Reynolds numbers
STORE_EVERY_TRAIN = 20   # snapshot stride  ->  101 snapshots per Re
STORE_EVERY_VAL = 25     # coarser stride for the validation sets (81 each)
STORE_EVERY_TEST = 10    # stride for the test Reynolds numbers (201 each)

#: Extrapolation distance of the test, as a fraction of the training span.
EXTRAP_FRAC = 1.0
#: The validation extrapolation sits nearer the range (held out).
VAL_EXTRAP_FRAC = 0.5
SEED = 0                 # RNG seed selecting the random interpolation test Re

SPLITS = ("train", "val_interp", "val_extrap", "test_interp", "test_extrap")


def viscosity(Re: float) -> float:
    """``nu = U R / Re`` with ``U = R = 1``."""
    return 1.0 / float(Re)


def solve_boussinesq(Re: float, store_every: int) -> Tuple[np.ndarray, dict]:
    """Snapshots ``[u; v; theta]`` for one Reynolds number (disk-cached)."""
    DATA.mkdir(parents=True, exist_ok=True)
    cache = cache_path(float(Re), int(store_every))
    if cache.exists():
        z = np.load(cache)
        return z["S"], dict(X=z["X"], Y=z["Y"], tspan=z["tspan"],
                            field_shape=(NX, NY), Re=float(Re),
                            nu=viscosity(Re))

    from boussinesq import generate_boussinesq   # noqa: E402

    S, info = generate_boussinesq(
        Nx=NX, Ny=NY, Lx=LX, Ly=LY, Re=float(Re), Pr=PR, dt=DT, t_end=T_END,
        store_every=int(store_every), y0=Y0, R=R,
    )
    np.savez(cache, S=S, X=info["X"], Y=info["Y"], tspan=info["tspan"])
    return S, dict(X=info["X"], Y=info["Y"], tspan=info["tspan"],
                   field_shape=(NX, NY), Re=float(Re), nu=info["nu"])


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


def all_reynolds() -> Dict[str, list]:
    """Every Reynolds number of every split, keyed by split."""
    return {
        "train": list(map(float, training_reynolds())),
        "val_interp": list(map(float, val_interp_reynolds())),
        "val_extrap": [val_extrap_reynolds()],
        "test_interp": [test_interp_reynolds()],
        "test_extrap": [test_extrap_reynolds()],
    }


def store_every(split: str) -> int:
    return {"train": STORE_EVERY_TRAIN, "val_interp": STORE_EVERY_VAL,
            "val_extrap": STORE_EVERY_VAL, "test_interp": STORE_EVERY_TEST,
            "test_extrap": STORE_EVERY_TEST}[split]


def _stack(reynolds_numbers, se: int) -> Tuple[np.ndarray, dict]:
    """Concatenate the raw snapshots of several Reynolds numbers (full ``d``)."""
    blocks, info = [], None
    for Re in np.atleast_1d(reynolds_numbers):
        S, info = solve_boussinesq(float(Re), se)
        blocks.append(S)
    return np.concatenate(blocks, axis=1), info


def make_dataset() -> Dict[str, dict]:
    """The five splits in the full state dimension.

    Every ``S`` is ``(d, n_split)`` float64 (about 2.9 GB in total).  All
    snapshot files must already exist (``generate_data.py``); nothing is
    recomputed here.
    """
    missing = [pth.name for *_, pth in cached_runs() if not pth.exists()]
    if missing:
        raise FileNotFoundError(f"missing cached runs {missing}; run "
                                "generate_data.py first")
    res = all_reynolds()
    out = {}
    for split in SPLITS:
        se = store_every(split)
        parts = [np.load(cache_path(float(Re), se))["S"] for Re in res[split]]
        S = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        del parts
        out[split] = dict(S=np.ascontiguousarray(S, dtype=np.float64),
                          Re=res[split], field_shape=(NX, NY))
        if len(res[split]) == 1:
            out[split]["nu"] = viscosity(res[split][0])
    out["_meta"] = dict(d=D, n_train=int(out["train"]["S"].shape[1]))
    return out


make_raw_dataset = make_dataset


def cache_path(Re: float, se: int) -> Path:
    return DATA / f"boussinesq_N{NX}x{NY}_Re{Re:.6g}_se{se}.npz"


def cached_runs() -> list:
    """``[(split, Re, cache_path), ...]`` for every file the splits need."""
    res = all_reynolds()
    return [(split, float(Re), cache_path(float(Re), store_every(split)))
            for split in SPLITS for Re in res[split]]


# ---- field helpers ----------------------------------------------------------


def fields(col: np.ndarray):
    """``(u, v, theta)`` as ``(NX, NY)`` arrays from one stacked snapshot."""
    col = np.asarray(col)
    NN = NX * NY
    return (col[:NN].reshape(NX, NY), col[NN:2 * NN].reshape(NX, NY),
            col[2 * NN:].reshape(NX, NY))


def speed(col: np.ndarray) -> np.ndarray:
    """Velocity magnitude ``sqrt(u^2 + v^2)`` of one stacked snapshot."""
    u, v, _ = fields(col)
    return np.sqrt(u ** 2 + v ** 2)


def temperature(col: np.ndarray) -> np.ndarray:
    return fields(col)[2]
