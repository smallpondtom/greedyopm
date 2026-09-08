"""NACA 0012 parametric problem: angle-of-attack splits and lossless compression.

The varied parameter is the **angle of attack** ``alpha`` (degrees) of the
NACA 0012 airfoil at fixed ``Re = 1000`` (the aerodynamic analogue of the
Reynolds-number sweep in the 2-D Burgers example).  Across the range the wake
goes from thin attached-shear-layer shedding to progressively deeper stall, so
no low-dimensional linear subspace superposes the family: the parametric
snapshot matrix has a slowly decaying Kolmogorov n-width (see ``sv_decay.py``),
which is the regime the Greedy Orthogonal Polynomial Manifold targets.

Parametric splits (train / validation / test), mirroring lin_adv_diff / burgers2d
--------------------------------------------------------------------------------
* ``train``        -- angles evenly on ``[ALPHA_MIN, ALPHA_MAX] = [4, 16] deg``.
* ``val_interp``   -- held-out interior angles (interpolation validation).
* ``val_extrap``   -- a held-out angle just beyond ``ALPHA_MAX`` (a nearer
  extrapolation than the extrapolation test) used to tune the budget.
* ``test_interp``  -- a held-out interior angle (interpolation test).
* ``test_extrap``  -- an angle in deep stall well beyond the training range
  (extrapolation test), where a loose polynomial manifold over-extrapolates and
  the Lipschitz budget earns its keep.

Compression
-----------
The full velocity state is ``d = 2 * ny * nx = 262144``.  Every split is
compressed once to a shared orthonormal basis ``Q`` built from the union of all
splits' developed-window snapshots (centered by the training mean).  Because
``Q`` is an isometry on that span, the reduced coordinates ``A = Q^T (S - c)``
carry the full-state velocity error exactly (projection residual ~1e-15), so the
methods can be fit and evaluated in the small ``K``-dimensional space while every
reported number remains a true full-state error.  This is a lossless
implementation detail; the burgers2d example, an order of magnitude smaller in
``d``, works in the full space directly.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(_HERE, "data")
COMP = os.path.join(_HERE, "data", "interp")

# ---- parameter regime: angle of attack (deg) --------------------------------

ALPHA_MIN, ALPHA_MAX = 4, 16
TRAIN_ANGLES = [4, 8, 12, 16]          # evenly on [ALPHA_MIN, ALPHA_MAX]
VAL_INTERP_ANGLES = [6, 14]            # held-out interior (interp. validation)
TEST_INTERP_ANGLE = 10                 # held-out interior (interp. test)
VAL_EXTRAP_ANGLE = 18                  # nearer held-out extrapolation
TEST_EXTRAP_ANGLE = 20                 # deep-stall extrapolation test

T_DEV = 15.0                           # developed (post-transient) window

SPLITS = ("train", "val_interp", "val_extrap", "test_interp", "test_extrap")
SPLIT_ANGLES = {
    "train": TRAIN_ANGLES,
    "val_interp": VAL_INTERP_ANGLES,
    "val_extrap": [VAL_EXTRAP_ANGLE],
    "test_interp": [TEST_INTERP_ANGLE],
    "test_extrap": [TEST_EXTRAP_ANGLE],
}
ALL_ANGLES = sorted({a for v in SPLIT_ANGLES.values() for a in v})

_CACHE: dict = {}


# ---- raw snapshot loading ---------------------------------------------------


def _raw_path(a: int) -> str:
    return os.path.join(RAW, f"naca_a{a:02d}.npz")


def _load_angle(a: int):
    """Developed-window stacked velocity ``[u; v]`` of one angle, ``(d, k)`` (f64).

    Loaded in float64 so the method-of-snapshots Gram matrix accumulates
    accurately over the large state dimension (float32 accumulation noise would
    swamp the tail singular values and corrupt the basis).
    """
    d = np.load(_raw_path(a))
    keep = d["t"] >= T_DEV
    u = d["u"][keep]
    v = d["v"][keep]
    k = u.shape[0]
    S = np.concatenate([u.reshape(k, -1), v.reshape(k, -1)], axis=1).T
    return np.ascontiguousarray(S, dtype=np.float64), d


# ---- lossless union-basis compression --------------------------------------


def _build_compression() -> None:
    missing = [a for a in ALL_ANGLES if not os.path.exists(_raw_path(a))]
    if missing:
        raise FileNotFoundError(
            f"missing raw data for angles {missing}; generate them first "
            f"(naca_solver.py at those NS_ALPHA)."
        )
    os.makedirs(COMP, exist_ok=True)
    # Memory-frugal method-of-snapshots (state is d=262144): keep only the
    # per-angle blocks (float32) and build the Gram matrix, basis, and reduced
    # coords block-by-block -- the full union matrix X is never materialized.
    import gc

    blocks, order_a, geo = {}, [], None
    train_sum, train_n = None, 0
    for a in ALL_ANGLES:
        S, d = _load_angle(a)
        blocks[a] = S
        order_a.append(a)
        if a in TRAIN_ANGLES:
            s = S.sum(axis=1, keepdims=True, dtype=np.float64)
            train_sum = s if train_sum is None else train_sum + s
            train_n += S.shape[1]
        if geo is None:
            geo = dict(x=d["x"], y=d["y"], chi=d["chi"],
                       ny=int(d["u"].shape[1]), nx=int(d["u"].shape[2]),
                       chord=float(d["chord"]), U_inf=float(d["U_inf"]))
    c = (train_sum / train_n)                             # training mean (d,1) f64
    for a in order_a:                                     # center in place
        blocks[a] -= c
    sizes = {a: blocks[a].shape[1] for a in order_a}
    offs, o = {}, 0
    for a in order_a:
        offs[a] = o
        o += sizes[a]
    N = o

    # Gram matrix G = X^T X (float64), one (n_i x n_j) block at a time.
    G = np.zeros((N, N), dtype=np.float64)
    for a in order_a:
        for b in order_a:
            if offs[b] < offs[a]:
                continue
            gij = blocks[a].T @ blocks[b]
            G[offs[a]:offs[a] + sizes[a], offs[b]:offs[b] + sizes[b]] = gij
            if a != b:
                G[offs[b]:offs[b] + sizes[b], offs[a]:offs[a] + sizes[a]] = gij.T
    w, V = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1]
    w = np.clip(w[idx], 0.0, None)
    V = V[:, idx]
    sig = np.sqrt(w)
    K = int(np.sum(sig > sig[0] * 1e-7))
    Vk = V[:, :K] / np.maximum(sig[:K], 1e-30)[None, :]   # (N, K) f64

    # Q = X Vk, accumulated block-by-block (X never formed), in float64.
    d_full = c.shape[0]
    Q = np.zeros((d_full, K), dtype=np.float64)
    for a in order_a:
        Q += blocks[a] @ Vk[offs[a]:offs[a] + sizes[a], :]
    del V, Vk, G
    gc.collect()

    # Reduced coords and lossless residual, per angle (no big temporaries).
    A, sq_res, sq_tot = {}, 0.0, 0.0
    for a in order_a:
        Aa = Q.T @ blocks[a]                              # (K, n_a) f64
        A[a] = Aa
        recon = Q @ Aa                                    # (d, n_a) f64
        sq_res += float(np.sum((blocks[a] - recon) ** 2))
        sq_tot += float(np.sum(blocks[a] ** 2))
        del recon
    resid = float(np.sqrt(sq_res / sq_tot))
    ortho = float(np.linalg.norm(Q.T @ Q - np.eye(K)))

    np.save(os.path.join(COMP, "Q.npy"), Q.astype(np.float32))
    np.save(os.path.join(COMP, "cmean.npy"), c.astype(np.float32))
    for a in order_a:
        np.save(os.path.join(COMP, f"A_a{a:02d}.npy"), A[a])
    counts = {a: (offs[a], sizes[a]) for a in order_a}
    del blocks, Q
    gc.collect()
    np.savez(os.path.join(COMP, "geom.npz"), **geo)
    meta = dict(K=int(K), d=int(d_full), ny=geo["ny"], nx=geo["nx"],
                proj_residual=resid, ortho_residual=ortho, T_dev=T_DEV,
                counts={int(a): int(n) for a, (off, n) in counts.items()},
                train_angles=TRAIN_ANGLES, val_interp=VAL_INTERP_ANGLES,
                val_extrap=VAL_EXTRAP_ANGLE, test_interp=TEST_INTERP_ANGLE,
                test_extrap=TEST_EXTRAP_ANGLE)
    json.dump(meta, open(os.path.join(COMP, "meta.json"), "w"), indent=2)


def _ensure() -> None:
    if not os.path.exists(os.path.join(COMP, "meta.json")):
        _build_compression()


def _A(a: int) -> np.ndarray:
    return np.load(os.path.join(COMP, f"A_a{a:02d}.npy"))


def make_dataset() -> Dict[str, dict]:
    """Return the reduced-coordinate splits (``S`` = compressed coords ``A``)."""
    _ensure()
    meta = json.load(open(os.path.join(COMP, "meta.json")))
    out = {}
    for sp in SPLITS:
        angles = SPLIT_ANGLES[sp]
        S = np.concatenate([_A(a) for a in angles], axis=1)
        out[sp] = dict(S=S, alpha=list(map(float, angles)),
                       n_per=[int(_A(a).shape[1]) for a in angles],
                       field_shape=(meta["ny"], meta["nx"]))
    out["_meta"] = meta
    return out


# ---- lifting to physical velocity (for field panels) ------------------------


def load_lifting():
    if "Q" not in _CACHE:
        _CACHE["Q"] = np.load(os.path.join(COMP, "Q.npy")).astype(np.float64)
        _CACHE["c"] = np.load(os.path.join(COMP, "cmean.npy")).astype(np.float64).ravel()
    return _CACHE["Q"], _CACHE["c"]


def load_geom() -> dict:
    if "geom" not in _CACHE:
        g = np.load(os.path.join(COMP, "geom.npz"))
        _CACHE["geom"] = dict(x=g["x"], y=g["y"], chi=g["chi"],
                              ny=int(g["ny"]), nx=int(g["nx"]),
                              chord=float(g["chord"]), U_inf=float(g["U_inf"]))
    return _CACHE["geom"]


def speed(a_col: np.ndarray) -> np.ndarray:
    """Lift a reduced-coordinate column to the physical speed field ``|u|``."""
    Q, c = load_lifting()
    g = load_geom()
    ny, nx = g["ny"], g["nx"]
    full = Q @ np.asarray(a_col).ravel() + c
    npix = ny * nx
    u = full[:npix].reshape(ny, nx)
    v = full[npix:].reshape(ny, nx)
    return np.sqrt(u ** 2 + v ** 2)
