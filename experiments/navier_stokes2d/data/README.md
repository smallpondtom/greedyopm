# data/

* `naca0012_lam.su2`, `mesh.npz`, `mesh_preview.png` -- the body-fitted hybrid mesh
  (`make_mesh.py`), shared by every angle.
* `naca_aXX.npz` -- one SU2 run per angle of attack (laminar, Re_c = 1000, t in
  [0, 60], snapshots every 0.2): `x, y, t, u, v, p, force_t, Cl, Cd, tris, quads,
  airfoil_nodes` plus scalars. About 120 MB each. Delivered in 19 MB pieces under
  `_chunks/`; run `python join_chunks.py` once to reassemble them, then delete
  `_chunks/`. Regenerate with `run_param.py` (needs SU2_CFD).
* `compressed/` -- built on first use by `problem.py` (union-basis compression).
* `archive_512x256_coarse/` -- the retired Fourier/penalization data (see its README).
