"""Recover the (x, y) coordinate of every interior velocity DOF, aligned with
the stored snapshot ordering, and save it to ``coords_stored.npy``.

Why this is needed
------------------
The snapshots in ``nse_snapshots.npz`` (and the mass matrix in
``simulation_nse/karman2D-outlets_lvl1_TH.npz``) were produced on another
machine with ``dolfin_navier_scipy``.  Legacy dolfin renumbers DOFs with a
build-dependent graph reordering, so ``tabulate_dof_coordinates`` from a fresh
dolfin install does NOT match the stored ordering (verified: the stored mass
matrix would then connect spatially distant DOFs).

However, the stored mass matrix M and a freshly assembled one are the *same*
weighted graph (identical nonzero values, just relabelled).  We therefore:

  1. rebuild the dns velocity space, assemble its interior mass matrix and read
     its DOF coordinates (self-consistent, known ordering);
  2. recover the relabelling with Weisfeiler-Lehman colour refinement on the two
     mass graphs (edge weight = the mass entry, used as a fingerprint);
  3. map the known coordinates through that relabelling.

The u/v components share identical graph neighbourhoods, so refinement converges
to one colour per spatial node; the interleaved layout ([u0, v0, u1, v1, ...],
verified on both matrices) lets us assign components by parity.

Run in the legacy-FEniCS env (has dolfin + dns + numpy/scipy):

    source ~/miniforge3/etc/profile.d/conda.sh && conda activate fenics2019
    PRE=$CONDA_PREFIX FI_PROVIDER=tcp \
      PKG_CONFIG_PATH=$PRE/lib/pkgconfig CMAKE_PREFIX_PATH=$PRE \
      $PRE/bin/python recover_dof_coords.py

Result is validated by the M-graph edge-length check at the end (all interior
mass-graph edges must be short, i.e. within a couple of cell diameters).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sps

_HERE = Path(__file__).resolve().parent
MESHDIR = _HERE / "data" / "mesh"
TH = MESHDIR / "karman2D-outlets_lvl1_TH.npz"  # stored interior mass matrix
QW = 9  # decimals to quantise mass-matrix weights for fingerprinting


def assemble_true():
    """dns interior vector mass matrix + DOF coords in a known dolfin ordering."""
    import dolfin
    import dolfin_navier_scipy.problem_setups as dnsps
    femp, *_ = dnsps.get_sysmats(
        problem="gen_bccont", Re=30, bccontrol=False, scheme="TH", mergerhs=True,
        meshparams=dict(strtomeshfile=str(MESHDIR / "karman2D-outlets_lvl1.xml.gz"),
                        strtophysicalregions=str(MESHDIR / "karman2D-outlets_lvl1_facet_region.xml.gz"),
                        strtobcsobs=str(MESHDIR / "karman2D-outlets_geo_cntrlbc.json")))
    V = femp["V"]; invinds = np.asarray(femp["invinds"]).ravel()
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)
    M = dolfin.as_backend_type(dolfin.assemble(dolfin.inner(u, v) * dolfin.dx)).sparray()
    Mint = M.tocsr()[invinds][:, invinds].tocsr()
    coords = V.tabulate_dof_coordinates().reshape(-1, 2)[invinds]
    return Mint, coords


def _refine(M, lab, sig_map):
    ipt, idx, w = M.indptr, M.indices, np.round(M.data, QW)
    new = np.empty(M.shape[0], np.int64)
    for i in range(M.shape[0]):
        s, e = ipt[i], ipt[i + 1]
        sig = (lab[i],) + tuple(sorted(zip(w[s:e].tolist(), lab[idx[s:e]].tolist())))
        x = sig_map.get(sig)
        if x is None:
            x = len(sig_map); sig_map[sig] = x
        new[i] = x
    return new


def main():
    Mt, ct = assemble_true()
    th = np.load(TH)
    Ms = sps.csr_matrix((th["data"], th["indices"], th["indptr"]),
                        shape=tuple(th["shape"])).tocsr()
    n = Mt.shape[0]
    assert Ms.shape == Mt.shape and Ms.nnz == Mt.nnz, "mass graphs differ"

    lab_t = np.zeros(n, np.int64); lab_s = np.zeros(n, np.int64)
    for _ in range(4):
        sm = {}
        lab_t = _refine(Mt, lab_t, sm)
        lab_s = _refine(Ms, lab_s, sm)

    # one colour per spatial node; interleaved [u, v] -> use even (u) indices
    true_scalar_of = {int(lab_t[2 * a]): a for a in range(n // 2)}
    coords = np.empty((n, 2)); miss = 0
    for k in range(n // 2):
        a = true_scalar_of.get(int(lab_s[2 * k]))
        if a is None:
            miss += 1; coords[2 * k] = coords[2 * k + 1] = np.nan; continue
        coords[2 * k] = coords[2 * k + 1] = ct[2 * a]
    print(f"unmatched spatial nodes: {miss} / {n // 2}")

    Mc = Ms.tocoo()
    d = np.linalg.norm(coords[Mc.row] - coords[Mc.col], axis=1)
    print(f"M-graph edge length: median={np.median(d):.5f} "
          f"95%={np.percentile(d, 95):.5f} max={d.max():.5f} "
          f"frac>0.2={(d > 0.2).mean():.5f}  (must be ~0)")
    assert miss == 0 and (d > 0.2).mean() == 0.0, "recovery failed"

    np.save(_HERE / "coords_stored.npy", coords)
    print(f"saved {_HERE / 'coords_stored.npy'}  {coords.shape}")


if __name__ == "__main__":
    main()
