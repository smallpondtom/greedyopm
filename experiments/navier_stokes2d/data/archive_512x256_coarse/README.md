# Archived coarse-grid NACA 0012 data (512 x 256, dt = 0.005, eps = 5e-4)

Generated 2026-08-30 with `naca_solver_as_run.py` (a verbatim copy of the
solver at the time) and compressed by `problem_as_run.py` into `interp/`.
These are the snapshots behind every figure in `paper/.../figures/ns/` and
in `plots/naca_*.json` as of 2026-09-12.

Archived on 2026-09-12 because a comparison against Kurtulus (2015,
Int. J. Micro Air Vehicles 7(3), 301-326) showed the flow is not a faithful
NACA 0012 at Re = 1000 on this grid: the airfoil is 4 cells thick, the wake
sheds at alpha = 4 deg where the reference flow is steady, and the wake
Strouhal number is 0.395 for alpha = 4..12 deg against St = 0.876 at 8 deg
in the reference.  Kept read-only for provenance; do not regenerate into
this folder.
