"""3-by-1 velocity-magnitude reconstruction panel for Greedy-OPM.

Renders, for one developed-shedding snapshot of the Karman vortex:

    (1) ground-truth |u|,  (2) Greedy-OPM reconstruction |u|,  (3) |error|

The unstructured P2 velocity vector (41682 interior DOFs) is placed in space
via ``dof_coords.npy`` (the (42764, 2) DOF coordinates of the CG2 vector space,
produced once with legacy dolfin), restricted to the interior DOFs through
``invinds`` and interpolated onto a fine regular grid with the cylinder / outside
masked, at the true (long) aspect ratio.

Run:  python nse_greedyopm_field.py [method] [snapshot_index] [r]
"""

from __future__ import annotations

import gzip
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import cKDTree
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
sys.path.insert(0, str(_HERE.parent))                      # experiments/
from exp_methods import build_model                        # noqa: E402

RAW = DATA_DIR / "nse_snapshots.npz"
# per-interior-DOF (x, y) aligned to the stored snapshot ordering, recovered by
# matching the stored mass matrix to a dolfin-assembled one (see recover_coords).
COORDS = DATA_DIR / "coords_stored.npy"
MESH = DATA_DIR / "mesh" / "karman2D-outlets_lvl1.xml.gz"
PLOTS = _HERE / "plots"

METHOD = "Greedy-OPM (p=[2,3])"
SNAP_INDEX = 1600          # global snapshot index (even -> train); t ~ 9.6
R_FIELD = 10
GRID_NX = 1100
CMAP = "viridis"
CFG = dict(
    s_pool=100, 
    gamma_qm=1e-12, 
    gamma_opm=1e-12, 
    gamma_geelen=1e-8,
    geelen_rel=0.10, 
    geelen_iters=400, 
    tau=1e-3
)


# ---------------------------------------------------------------------------
# unstructured P2 field -> velocity magnitude on the FEM mesh vertices
# ---------------------------------------------------------------------------

def read_mesh(path):
    """Vertices (Nv, 2) and triangles (Nt, 3) of the dolfin XML mesh."""
    mesh = ET.parse(gzip.open(path)).getroot().find("mesh")
    vs = mesh.find("vertices")
    verts = np.empty((int(vs.get("size")), 2))
    for v in vs:
        verts[int(v.get("index"))] = (float(v.get("x")), float(v.get("y")))
    cs = mesh.find("cells")
    tris = np.empty((int(cs.get("size")), 3), dtype=int)
    for c in cs:
        tris[int(c.get("index"))] = (int(c.get("v0")), int(c.get("v1")),
                                     int(c.get("v2")))
    return verts, tris


def nodes_from_coords(coords_interior):
    """Group interior DOF coordinates into spatial nodes (u, v share a point)."""
    key = np.round(coords_interior, 9)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    nnode = int(inv.max()) + 1
    node_xy = np.zeros((nnode, 2))
    node_xy[inv] = coords_interior
    return inv, nnode, node_xy


def node_speed(vec_interior, inv, nnode):
    """Velocity magnitude per spatial node from the per-DOF interior vector."""
    sq = np.zeros(nnode)
    np.add.at(sq, inv, np.asarray(vec_interior) ** 2)      # sum u^2 + v^2
    return np.sqrt(sq)


def vertex_speed(spd_node, node_xy, verts):
    """Map per-node speed to the mesh vertices; fill boundary vertices (whose
    DOFs were removed by the Dirichlet conditions) from the nearest node."""
    tree = cKDTree(node_xy)
    d, idx = tree.query(verts)
    vs = spd_node[idx]                       # nearest-node value (exact when d~0)
    return vs


# ---------------------------------------------------------------------------

def main(method=METHOD, g=SNAP_INDEX, r=R_FIELD):
    if not COORDS.exists():
        sys.exit(f"missing {COORDS}")
    coords_int = np.load(COORDS)             # (41682, 2) in stored DOF ordering
    raw = np.load(RAW)
    V = raw["V"].astype(np.float64)          # (41682, 2000) interior velocity
    trange = raw["trange"]
    inv, nnode, node_xy = nodes_from_coords(coords_int)
    verts, tris = read_mesh(MESH)
    triang = mtri.Triangulation(verts[:, 0], verts[:, 1], tris)
    extent = [verts[:, 0].min(), verts[:, 0].max(),
              verts[:, 1].min(), verts[:, 1].max()]
    print(f"[field] {nnode} interior nodes, {len(verts)} vertices, "
          f"{len(tris)} triangles, extent {extent}")

    # ---- lossless training POD basis (method of snapshots), matches prep_nse
    Vtr = V[:, ::2]
    s_ref = Vtr.mean(axis=1, keepdims=True)
    Xtr = Vtr - s_ref
    w, W = np.linalg.eigh(Xtr.T @ Xtr)
    w = np.clip(w[::-1], 0, None); W = W[:, ::-1]
    keep = np.sqrt(w) > np.sqrt(w[0]) * np.sqrt(1e-13)
    K = int(keep.sum()); sig = np.sqrt(w[:K]); W = W[:, :K]
    Q = Xtr @ (W / sig)                                    # (d_int, K), orthonormal
    A_train = sig[:, None] * W.T                           # (K, n_train)
    print(f"[field] K={K}")

    # ---- reduced coordinate of snapshot g and Greedy-OPM reconstruction
    truth_int = V[:, g]
    a_col = Q.T @ (truth_int - s_ref[:, 0])                # (K,)
    print(f"[field] building {method} at r={r} ...")
    model = build_model(method, jnp.asarray(A_train), r, CFG)
    ahat = np.asarray(model.reconstruct(jnp.asarray(a_col[:, None])))[:, 0]
    rec_int = s_ref[:, 0] + Q @ ahat

    rel = np.linalg.norm(rec_int - truth_int) / np.linalg.norm(truth_int)
    print(f"[field] snapshot {g} (t={trange[g]:.2f})  full-state rel.err={rel:.3e}")

    truth_v = vertex_speed(node_speed(truth_int, inv, nnode), node_xy, verts)
    rec_v = vertex_speed(node_speed(rec_int, inv, nnode), node_xy, verts)
    err_v = rec_v - truth_v                                 # signed error

    # ---- 3 x 1 figure on the true FEM triangulation
    aspect_w = (extent[1] - extent[0]) / (extent[3] - extent[2])
    fig, axes = plt.subplots(3, 1, figsize=(13, 3 * 13 / aspect_w * 1.18))
    vmin, vmax = float(truth_v.min()), float(truth_v.max())
    speed_lv = np.linspace(vmin, vmax, 200)
    emax = float(np.abs(err_v).max()) or 1.0
    err_lv = np.linspace(-emax, emax, 200)                  # symmetric about 0

    def panel(ax, vals, cmap, levels, title):
        tcf = ax.tricontourf(triang, vals, levels=levels, cmap=cmap, extend="both")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(tcf, ax=ax, fraction=0.018, pad=0.01)
        cb.set_ticks(np.linspace(levels[0], levels[-1], 5))

    panel(axes[0], truth_v, CMAP, speed_lv, "Ground truth  $|u|$")
    panel(axes[1], rec_v, CMAP, speed_lv,
          f"{method} reconstruction  $|\\hat u|$  (r={r})")
    panel(axes[2], err_v, "RdBu_r", err_lv,
          "Error  $|\\hat u| - |u|$")
    fig.suptitle(f"Karman vortex — Greedy-OPM velocity-magnitude reconstruction "
                 f"(t={trange[g]:.2f}, rel.err={rel:.2e})", fontsize=16, y=1.0)
    plt.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "nse_greedyopm_field"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out}.png/.pdf")


if __name__ == "__main__":
    a = sys.argv
    main(a[1] if len(a) > 1 else METHOD,
         int(a[2]) if len(a) > 2 else SNAP_INDEX,
         int(a[3]) if len(a) > 3 else R_FIELD)
