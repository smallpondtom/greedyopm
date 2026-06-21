"""Wide 2D velocity-field reconstruction panels for the Karman vortex.

Renders ground truth and every method's reconstruction (plus |error|) for a
chosen snapshot, interpolating the unstructured FEM velocity magnitude onto a
fine regular grid with the cylinder/outside masked, at the true (long) aspect
ratio.

Requires the DOF coordinates of the velocity space, aligned with the stored
42764-vector, in ``transient_ns_pbenner/dof_coords.npy`` (shape (42764, 2)).
Produce it once in a dolfin session that has the mesh:

    import dolfin, numpy as np
    mesh = dolfin.Mesh('.../karman2D-outlets_lvl1.xml.gz')
    V = dolfin.VectorFunctionSpace(mesh, 'CG', 2)
    np.save('dof_coords.npy', V.tabulate_dof_coordinates().reshape(-1, 2))

(The mesh alone is insufficient: dolfin reorders DOFs, so the value<->coordinate
map can only be obtained from the live function space.)

Run ``python nse_field.py selftest`` to exercise the interpolation/masking on
synthetic data without the real coordinates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
SRC = Path("/sessions/loving-gallant-allen/mnt/transient_ns_pbenner")
RAW = SRC / "nse_snapshots.npz"
COORDS = SRC / "dof_coords.npy"
PLOTS = _HERE / "plots"

R_FIELD = 10
SNAP_INDEX = 1600          # global snapshot index (t ~ 9.6, developed shedding)
GRID_NX = 900
CMAP = "coolwarm"


# ---------------------------------------------------------------------------
# unstructured field -> masked regular grid
# ---------------------------------------------------------------------------

def nodes_from_coords(coords_interior):
    """Group the interior DOF coordinates into spatial nodes (u,v share a point)."""
    key = np.round(coords_interior, 9)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    nnode = inv.max() + 1
    node_xy = np.zeros((nnode, 2))
    node_xy[inv] = coords_interior
    return inv, nnode, node_xy


def node_speed(vec_interior, inv, nnode):
    """Velocity magnitude per spatial node from the per-DOF interior vector."""
    sq = np.zeros(nnode)
    np.add.at(sq, inv, vec_interior**2)        # sum u^2 + v^2 at each node
    return np.sqrt(sq)


def to_grid(node_xy, vals, extent, nx, mask_factor=2.2):
    xmin, xmax, ymin, ymax = extent
    ny = max(int(nx * (ymax - ymin) / (xmax - xmin)), 40)
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(gx, gy)
    grid = griddata(node_xy, vals, (GX, GY), method="linear")
    # mask the cylinder hole and outside-domain via nearest-node distance
    tree = cKDTree(node_xy)
    d, _ = tree.query(np.c_[GX.ravel(), GY.ravel()])
    nn = np.median(tree.query(node_xy, k=2)[0][:, 1])     # typical node spacing
    grid = np.ma.array(grid.reshape(GX.shape),
                       mask=(d.reshape(GX.shape) > mask_factor * nn)
                       | ~np.isfinite(grid.reshape(GX.shape)))
    return grid


def panel_figure(truth_img, recon_imgs, names, extent, out_prefix, suptitle):
    n = len(names)
    aspect_w = (extent[1] - extent[0]) / (extent[3] - extent[2])
    fig, axes = plt.subplots(n + 1, 2, figsize=(2 * 5.2, (n + 1) * 5.2 / aspect_w * 1.15))
    vmin, vmax = float(np.nanmin(truth_img)), float(np.nanmax(truth_img))
    errs = {nm: np.ma.abs(recon_imgs[nm] - truth_img) for nm in names}
    emax = max(float(np.nanmax(errs[nm])) for nm in names) or 1.0
    ikw = dict(extent=extent, origin="lower", aspect="equal")

    a = axes[0, 0]
    im = a.imshow(truth_img, cmap=CMAP, vmin=vmin, vmax=vmax, **ikw)
    a.set_title("Ground truth  |u|", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=a, fraction=0.025, pad=0.01)
    axes[0, 1].axis("off")
    for i, nm in enumerate(names, 1):
        a0, a1 = axes[i, 0], axes[i, 1]
        im = a0.imshow(recon_imgs[nm], cmap=CMAP, vmin=vmin, vmax=vmax, **ikw)
        a0.set_title(nm, fontsize=12)
        fig.colorbar(im, ax=a0, fraction=0.025, pad=0.01)
        ie = a1.imshow(errs[nm], cmap="inferno", vmin=0, vmax=emax, **ikw)
        a1.set_title(f"|error|  ({nm})", fontsize=12)
        fig.colorbar(ie, ax=a1, fraction=0.025, pad=0.01)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(suptitle, fontsize=16, y=1.005)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_prefix}.png/.pdf")


# ---------------------------------------------------------------------------
# real run
# ---------------------------------------------------------------------------

def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    sys.path.insert(0, str(_HERE.parent))
    from exp_methods import build_model, METHOD_NAMES

    if not COORDS.exists():
        print(f"[field] MISSING {COORDS}\n"
              "        Add dof_coords.npy (see header) and rerun.")
        return
    full_coords = np.load(COORDS)
    raw = np.load(RAW)
    V = raw["V"]; invinds = raw["invinds"]; trange = raw["trange"]
    coords_int = full_coords[invinds]
    inv, nnode, node_xy = nodes_from_coords(coords_int)
    extent = [node_xy[:, 0].min(), node_xy[:, 0].max(),
              node_xy[:, 1].min(), node_xy[:, 1].max()]
    print(f"[field] {nnode} spatial nodes, extent {extent}")

    # train POD basis Q (interior x K) via method of snapshots
    Vtr = V[:, ::2].astype(np.float64)
    s_ref = Vtr.mean(axis=1, keepdims=True)
    Xtr = Vtr - s_ref
    w, W = np.linalg.eigh(Xtr.T @ Xtr)
    w = np.clip(w[::-1], 0, None); W = W[:, ::-1]
    keep = np.sqrt(w) > np.sqrt(w[0]) * np.sqrt(1e-13)
    K = int(keep.sum()); sig = np.sqrt(w[:K]); W = W[:, :K]
    Q = Xtr @ (W / sig)                              # (d_int, K)
    A_train = sig[:, None] * W.T                     # compressed train coords

    g = SNAP_INDEX
    truth_int = V[:, g].astype(np.float64)
    a_col = Q.T @ (truth_int - s_ref[:, 0])          # compressed coord of snapshot g

    recon = {}
    for nm in METHOD_NAMES:
        model = build_model(nm, jnp.asarray(A_train), R_FIELD,
                            dict(s_pool=25, gamma_qm=1e-12, gamma_opm=1e-12,
                                 gamma_geelen=1e-8, geelen_rel=0.10,
                                 geelen_iters=400, tau=1e-6))
        ahat = np.asarray(model.reconstruct(jnp.asarray(a_col[:, None])))[:, 0]
        rec_int = s_ref[:, 0] + Q @ ahat
        recon[nm] = to_grid(node_xy, node_speed(rec_int, inv, nnode), extent, GRID_NX)

    truth_img = to_grid(node_xy, node_speed(truth_int, inv, nnode), extent, GRID_NX)
    PLOTS.mkdir(parents=True, exist_ok=True)
    panel_figure(truth_img, recon, METHOD_NAMES, extent,
                 str(PLOTS / "nse_field_snapshot"),
                 f"Karman vortex — velocity magnitude reconstruction "
                 f"(t={trange[g]:.2f}, r={R_FIELD})")
    print("DONE")


def selftest():
    """Exercise interpolation + masking on synthetic channel+cylinder data."""
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [2.2, 0.41], size=(8000, 2))
    cc, cr = np.array([0.3, 0.2]), 0.06
    pts = pts[np.linalg.norm(pts - cc, axis=1) > cr]      # carve cylinder hole
    speed = (4 * pts[:, 1] * (0.41 - pts[:, 1]) / 0.41**2
             + 0.3 * np.sin(6 * pts[:, 0]) * np.exp(-((pts[:, 1] - 0.2)**2) / 0.02))
    ext = [0, 2.2, 0, 0.41]
    truth = to_grid(pts, speed, ext, 700)
    recon = {"A": to_grid(pts, speed * 0.9, ext, 700),
             "B": to_grid(pts, speed + 0.05 * rng.standard_normal(speed.shape), ext, 700)}
    PLOTS.mkdir(parents=True, exist_ok=True)
    panel_figure(truth, recon, ["A", "B"], ext, str(PLOTS / "nse_field_selftest"),
                 "selftest")
    print("selftest OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        main()
