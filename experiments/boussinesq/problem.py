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

Compression (lossless)
----------------------
``d = 344064`` is too large for the repeated ``d``-dimensional fits of the
greedy loop, so, as in the NACA experiment, every split is compressed once
to the orthonormal basis ``Q`` of the span of *all* splits' snapshots
(centered by the training mean ``c``).  With ``X = [S_1 - c, ..., S_m - c]``
and the eigendecomposition ``X^T X = V diag(sigma^2) V^T`` (method of
snapshots, float64), ``Q = X V diag(1/sigma)`` and the reduced coordinates
are ``A = Q^T X = diag(sigma) V^T`` -- so ``Q`` never has to be formed.
``Q`` is an isometry on the span (the discarded eigenvalues are at
round-off), so fitting and evaluating on ``A`` reports exact full-state
errors; ``Q`` is a change of coordinates, never part of any decoder, and
only the training split enters any fit.  :func:`lift` maps compressed
columns back to ``(u, v, theta)`` by streaming through the cached files,
``x = c + X (V diag(1/sigma) a)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data"

# ---- problem configuration --------------------------------------------------

NX, NY = 256, 448        # grid (state dimension d = 3 NX NY = 344064)
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
    cache = DATA / f"boussinesq_N{NX}x{NY}_Re{Re:.6g}_se{store_every}.npz"
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


def make_raw_dataset() -> Dict[str, dict]:
    """The five splits in full state dimension (memory heavy, ~3 GB)."""
    res = all_reynolds()
    out = {}
    for split in SPLITS:
        S, info = _stack(res[split], store_every(split))
        out[split] = dict(S=S, X=info["X"], Y=info["Y"],
                          t=np.asarray(info["tspan"]), Re=res[split],
                          field_shape=(NX, NY))
    return out


# ---- lossless union-basis compression ---------------------------------------

COMP = DATA / "compressed"
SV_TOL = 1e-7            # keep sigma_i > SV_TOL * sigma_0 (lossless in practice)


def _blocks() -> list:
    """``[(split, Re, cache_path), ...]`` in the fixed union ordering."""
    res = all_reynolds()
    out = []
    for split in SPLITS:
        se = store_every(split)
        for Re in res[split]:
            out.append((split, float(Re),
                        DATA / f"boussinesq_N{NX}x{NY}_Re{Re:.6g}_se{se}.npz"))
    return out


def _comp_key(split: str, Re: float) -> str:
    return f"{split}_Re{Re:.6g}"


def _build_compression(log=print) -> None:
    import json
    import time

    blocks = _blocks()
    missing = [p.name for *_, p in blocks if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing cached runs {missing}; run "
                                "generate_data.py first")
    COMP.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("[compress] loading blocks ...", flush=True)
    X = {i: np.load(p)["S"] for i, (*_, p) in enumerate(blocks)}
    n_tr = sum(X[i].shape[1] for i, (sp, *_) in enumerate(blocks) if sp == "train")
    c = sum(X[i].sum(axis=1, keepdims=True) for i, (sp, *_) in enumerate(blocks)
            if sp == "train") / n_tr
    for i in X:
        X[i] -= c
    counts = [X[i].shape[1] for i in range(len(blocks))]
    offs = np.concatenate([[0], np.cumsum(counts)])
    n = int(offs[-1])
    log(f"[compress] d={X[0].shape[0]} n_union={n} ({time.time() - t0:.0f}s), "
        "Gram matrix ...", flush=True)
    G = np.zeros((n, n))
    for i in range(len(blocks)):
        for j in range(i, len(blocks)):
            Gij = X[i].T @ X[j]
            G[offs[i]:offs[i + 1], offs[j]:offs[j + 1]] = Gij
            G[offs[j]:offs[j + 1], offs[i]:offs[i + 1]] = Gij.T
    del X
    w, V = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1]
    w = np.clip(w[idx], 0.0, None)
    V = V[:, idx]
    sig = np.sqrt(w)
    K = int(np.sum(sig > sig[0] * SV_TOL))
    A = sig[:K, None] * V[:, :K].T                        # = Q^T X, (K, n)
    resid = float(np.sqrt(w[K:].sum() / w.sum()))          # ||X - Q A|| / ||X||
    for i, (sp, Re, _) in enumerate(blocks):
        np.save(COMP / f"A_{_comp_key(sp, Re)}.npy", A[:, offs[i]:offs[i + 1]])
    np.save(COMP / "V.npy", V[:, :K])
    np.save(COMP / "sigma.npy", sig[:K])
    np.save(COMP / "cmean.npy", c)
    meta = dict(K=K, d=int(c.shape[0]), n_union=n, proj_residual=resid,
                sv_tol=SV_TOL, sigma_ratio_first_dropped=(
                    float(sig[K] / sig[0]) if K < n else None),
                blocks=[dict(split=sp, Re=Re, n=int(cnt))
                        for (sp, Re, _), cnt in zip(blocks, counts)],
                config=dict(NX=NX, NY=NY, RE_MIN=RE_MIN, RE_MAX=RE_MAX,
                            N_PARAMS_TRAIN=N_PARAMS_TRAIN, SEED=SEED,
                            EXTRAP_FRAC=EXTRAP_FRAC,
                            VAL_EXTRAP_FRAC=VAL_EXTRAP_FRAC,
                            STORE=[STORE_EVERY_TRAIN, STORE_EVERY_VAL,
                                   STORE_EVERY_TEST]),
                secs=round(time.time() - t0, 1))
    (COMP / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[compress] K={K} of {n}, projection residual {resid:.2e} "
        f"({time.time() - t0:.0f}s)", flush=True)


def _ensure_compression() -> dict:
    import json

    meta_path = COMP / "meta.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        want = [dict(split=sp, Re=Re) for sp, Re, _ in _blocks()]
        have = [dict(split=b["split"], Re=b["Re"]) for b in m.get("blocks", [])]
        if have == want and m.get("config", {}).get("NX") == NX \
                and m["config"].get("STORE") == [STORE_EVERY_TRAIN,
                                                 STORE_EVERY_VAL,
                                                 STORE_EVERY_TEST]:
            return m
    _build_compression()
    return json.loads(meta_path.read_text())


def make_dataset() -> Dict[str, dict]:
    """The five splits in compressed coordinates (``S`` = ``A = Q^T (S - c)``).

    Every ``S`` is ``(K, n_split)``; errors computed on these are exact
    full-state errors (see the module docstring).
    """
    meta = _ensure_compression()
    res = all_reynolds()
    out = {}
    for split in SPLITS:
        parts = [np.load(COMP / f"A_{_comp_key(split, Re)}.npy")
                 for Re in res[split]]
        out[split] = dict(S=np.concatenate(parts, axis=1), Re=res[split],
                          n_per=[int(p.shape[1]) for p in parts],
                          field_shape=(NX, NY))
        if len(res[split]) == 1:
            out[split]["nu"] = viscosity(res[split][0])
    out["_meta"] = meta
    return out


_LIFT_CACHE: dict = {}


def lift(cols: np.ndarray) -> np.ndarray:
    """Map compressed columns ``(K, m)`` to full states ``(d, m)``.

    ``x = c + X w`` with ``w = V diag(1/sigma) a``; the union snapshot matrix
    ``X`` is streamed block by block from the cached files, so no ``d x K``
    array is ever formed.
    """
    if "V" not in _LIFT_CACHE:
        _LIFT_CACHE["V"] = np.load(COMP / "V.npy")
        _LIFT_CACHE["sigma"] = np.load(COMP / "sigma.npy")
        _LIFT_CACHE["c"] = np.load(COMP / "cmean.npy")
    V, sig, c = (_LIFT_CACHE[k] for k in ("V", "sigma", "c"))
    a = np.atleast_2d(np.asarray(cols, dtype=np.float64))
    if a.shape[0] != V.shape[1]:
        a = a.T
    W = V @ (a / sig[:, None])                              # (n_union, m)
    out = np.repeat(c, a.shape[1], axis=1)
    off = 0
    for _, _, p in _blocks():
        Xb = np.load(p)["S"]
        nb = Xb.shape[1]
        out += (Xb - c) @ W[off:off + nb]
        off += nb
    return out


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
