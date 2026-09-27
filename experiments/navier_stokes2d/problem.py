"""NACA 0012 parametric problem (SU2 laminar data): angle-of-attack splits.

The data are laminar, unsteady SU2 solutions of the incompressible
Navier-Stokes equations past a NACA 0012 airfoil at ``Re_c = 1000`` on a
body-fitted hybrid mesh (``make_mesh.py``), one run per angle of attack
``alpha`` (``run_param.py``).  Across the range the wake goes from the onset of
shedding (alpha = 8 deg is the first unsteady angle in Kurtulus 2015) through
thin shear-layer shedding to progressively deeper stall, so no low-dimensional linear subspace superposes
the parametric family: the snapshot matrix has a slowly decaying Kolmogorov
n-width (see ``sv_decay.py``), the regime the Greedy Orthogonal Polynomial
Manifold targets.

Parametric splits (train / validation / test), mirroring lin_adv_diff / burgers2d
--------------------------------------------------------------------------------
* ``train``        -- angles evenly on ``[ALPHA_MIN, ALPHA_MAX] = [8, 20] deg``.
* ``val_interp``   -- held-out interior angles (interpolation validation).
* ``val_extrap``   -- a held-out angle just beyond ``ALPHA_MAX`` (a nearer
  extrapolation than the extrapolation test) used to tune the budget.
* ``test_interp``  -- a held-out interior angle (interpolation test).
* ``test_extrap``  -- an angle in deep stall well beyond the training range
  (extrapolation test), where a loose polynomial manifold over-extrapolates and
  the Lipschitz budget earns its keep.

State and norm
--------------
The modelled state stacks both velocity components on the ``N`` mesh nodes,
``d = 2 N``.  Because the mesh is strongly non-uniform (a fine boundary layer,
a refined wake, a coarse far field), every node is weighted by the square root
of its dual-cell area, ``s = W^{1/2} [u; v]``, so that the Euclidean norm of
``s`` is the discrete ``L^2(Omega)`` norm of the velocity field and all methods
optimize and are scored in that norm.  ``lift`` undoes the weighting for
plotting.  Only the developed window ``t >= T_DEV`` of each run is used.

Compression
-----------
Every split is compressed once to a shared orthonormal basis ``Q`` built from
the union of all splits' developed-window snapshots (centered by the training
mean).  ``Q`` is an isometry on that span (projection residual ~1e-12), so the
reduced coordinates ``A = Q^T (S - c)`` carry the full-state error exactly and
the methods are fit and evaluated in the small ``K``-dimensional space while
every reported number remains a true full-state error.  This is a lossless
implementation detail that keeps memory and run time small; ``lift`` maps a
compressed column back to the nodal velocity.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(_HERE, "data")
COMP = os.path.join(_HERE, "data", "compressed")

# ---- parameter regime: angle of attack (deg) --------------------------------

ALPHA_MIN, ALPHA_MAX = 8, 20
TRAIN_ANGLES = [8, 12, 16, 20]         # evenly on [ALPHA_MIN, ALPHA_MAX]
VAL_INTERP_ANGLES = [10, 18]           # held-out interior (interp. validation)
TEST_INTERP_ANGLE = 14                 # held-out interior (interp. test)
VAL_EXTRAP_ANGLE = 22                  # nearer held-out extrapolation
TEST_EXTRAP_ANGLE = 24                 # deep-stall extrapolation test

T_DEV = 30.0                           # developed (post-transient) window

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


# ---- mesh and quadrature weights --------------------------------------------


def _raw_path(a: int) -> str:
    return os.path.join(RAW, f"naca_a{a:02d}.npz")


def load_mesh() -> dict:
    """Node coordinates, connectivity, airfoil loop and dual-cell areas."""
    if "mesh" in _CACHE:
        return _CACHE["mesh"]
    m = np.load(os.path.join(RAW, "mesh.npz"))
    x, y = m["x"].astype(np.float64), m["y"].astype(np.float64)
    tris, quads, airfoil = m["tris"], m["quads"], m["airfoil_nodes"]
    area = np.zeros(len(x))
    for conn in (tris, quads):
        if len(conn) == 0:
            continue
        px, py = x[conn], y[conn]
        a = 0.5 * np.abs(np.sum(px * np.roll(py, -1, axis=1) - np.roll(px, -1, axis=1) * py, axis=1))
        share = a / conn.shape[1]
        for k in range(conn.shape[1]):
            np.add.at(area, conn[:, k], share)
    # quads split into two triangles for tri-based plotting
    tri_plot = np.vstack([tris, quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]) if len(quads) else tris
    geo = dict(x=x, y=y, tris=tris, quads=quads, tri_plot=tri_plot, airfoil=airfoil,
               area=area, sqrt_w=np.sqrt(area), n_nodes=len(x))
    _CACHE["mesh"] = geo
    return geo


# ---- snapshot loading -------------------------------------------------------


def _load_angle(a: int) -> np.ndarray:
    """Developed-window weighted state ``W^{1/2} [u; v]`` of one angle, ``(d, k)``."""
    d = np.load(_raw_path(a))
    keep = d["t"] >= T_DEV
    u = d["u"][keep].astype(np.float64)
    v = d["v"][keep].astype(np.float64)
    w = load_mesh()["sqrt_w"]
    S = np.concatenate([u * w, v * w], axis=1).T          # (2N, k)
    return np.ascontiguousarray(S)


def _build_compression() -> None:
    """Union-basis compression of all splits (method of snapshots, float64)."""
    missing = [a for a in ALL_ANGLES if not os.path.exists(_raw_path(a))]
    if missing:
        raise FileNotFoundError(
            f"missing SU2 data for angles {missing}; run run_param.py first.")
    os.makedirs(COMP, exist_ok=True)
    blocks = {a: _load_angle(a) for a in ALL_ANGLES}
    n_tr = sum(blocks[a].shape[1] for a in TRAIN_ANGLES)
    c = sum(blocks[a].sum(axis=1, keepdims=True) for a in TRAIN_ANGLES) / n_tr
    for a in ALL_ANGLES:
        blocks[a] -= c
    X = np.concatenate([blocks[a] for a in ALL_ANGLES], axis=1)
    G = X.T @ X
    w, V = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1]
    w = np.clip(w[idx], 0.0, None); V = V[:, idx]
    sig = np.sqrt(w)
    K = int(np.sum(sig > sig[0] * 1e-7))
    Q = X @ (V[:, :K] / np.maximum(sig[:K], 1e-30)[None, :])
    A = Q.T @ X
    resid = float(np.linalg.norm(X - Q @ A) / np.linalg.norm(X))
    ortho = float(np.linalg.norm(Q.T @ Q - np.eye(K)))
    np.save(os.path.join(COMP, "Q.npy"), Q)
    np.save(os.path.join(COMP, "cmean.npy"), c)
    off = 0
    for a in ALL_ANGLES:
        n_a = blocks[a].shape[1]
        np.save(os.path.join(COMP, f"A_a{a:02d}.npy"), A[:, off:off + n_a]); off += n_a
    meta = dict(K=int(K), d=int(X.shape[0]), n_union=int(X.shape[1]),
                proj_residual=resid, ortho_residual=ortho, T_dev=T_DEV,
                counts={int(a): int(blocks[a].shape[1]) for a in ALL_ANGLES},
                train_angles=TRAIN_ANGLES, val_interp=VAL_INTERP_ANGLES,
                val_extrap=VAL_EXTRAP_ANGLE, test_interp=TEST_INTERP_ANGLE,
                test_extrap=TEST_EXTRAP_ANGLE)
    json.dump(meta, open(os.path.join(COMP, "meta.json"), "w"), indent=2)


def _ensure() -> None:
    meta_path = os.path.join(COMP, "meta.json")
    if os.path.exists(meta_path):
        m = json.load(open(meta_path))
        same = (m.get("train_angles") == TRAIN_ANGLES and m.get("T_dev") == T_DEV
                and sorted(int(k) for k in m.get("counts", {})) == ALL_ANGLES)
        if same:
            return
    _build_compression()


def _A(a: int) -> np.ndarray:
    return np.load(os.path.join(COMP, f"A_a{a:02d}.npy"))


def make_dataset() -> Dict[str, dict]:
    """Return the compressed splits (``S`` = reduced coordinates ``A``)."""
    _ensure()
    meta = json.load(open(os.path.join(COMP, "meta.json")))
    geo = load_mesh()
    out = {}
    for sp in SPLITS:
        angles = SPLIT_ANGLES[sp]
        S = np.concatenate([_A(a) for a in angles], axis=1)
        out[sp] = dict(S=S, alpha=list(map(float, angles)),
                       n_per=[int(_A(a).shape[1]) for a in angles],
                       n_nodes=geo["n_nodes"])
    out["_meta"] = dict(meta, n_nodes=int(geo["n_nodes"]))
    return out


# ---- lifting to physical velocity (for field panels) ------------------------


def load_lifting():
    if "Q" not in _CACHE:
        _CACHE["Q"] = np.load(os.path.join(COMP, "Q.npy"))
        _CACHE["c"] = np.load(os.path.join(COMP, "cmean.npy")).ravel()
    return _CACHE["Q"], _CACHE["c"]


def lift(col: np.ndarray):
    """Map a compressed column to the nodal velocity ``(u, v)`` (weighting undone)."""
    Q, c = load_lifting()
    geo = load_mesh()
    full = (Q @ np.asarray(col).ravel() + c) / np.concatenate([geo["sqrt_w"], geo["sqrt_w"]])
    n = geo["n_nodes"]
    return full[:n], full[n:]


def speed(col: np.ndarray) -> np.ndarray:
    """Velocity magnitude ``|u|`` on the mesh nodes of one compressed column."""
    u, v = lift(col)
    return np.sqrt(u ** 2 + v ** 2)


def load_geom() -> dict:
    return load_mesh()
