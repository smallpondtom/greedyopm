# 2D NACA 0012 experiment (interpolation + extrapolation over angle of attack)

Comparison of the budget-calibrated Greedy-OPM against POD, Greedy-QM, Greedy-CM,
and Geelen-QM on the laminar, unsteady, incompressible flow past a NACA 0012
airfoil at \(Re_c = 1000\), with the **angle of attack** \(\alpha\) as the varied
parameter (the aerodynamic analogue of the Reynolds-number sweep in the 2-D
Burgers example). Across the range the flow goes from steady attached flow
(\(\alpha < 8^\circ\)) through thin shear-layer shedding to deep stall, so no
low-dimensional linear subspace superposes the family (`sv_decay.py`).

## Data: SU2, body-fitted mesh, validated against Kurtulus (2015)

The data are produced by **SU2** (v8.5, `INC_NAVIER_STOKES`, no turbulence
model) on a body-fitted hybrid mesh built by `make_mesh.py`:

* 320 surface points (cosine clustering), a 22-layer structured quad boundary
  layer with first cell \(0.002c\) and growth ratio 1.12 (thickness \(0.19c\)),
  a fan of cells at the sharp trailing edge, triangles outside, refinement boxes
  over the wake (\(h = 0.035c\) to \(x = 4c\), \(0.06c\) to \(x = 9c\), covering
  wake deflections up to \(\sim 25^\circ\)), far field a circle of radius \(30c\);
  37 572 nodes, 67 376 elements. One mesh serves every angle: \(\alpha\) enters
  through the free-stream direction, so the node ordering is identical across
  the sweep.
* Numerics (`su2_lam_config.py`): FDS convective flux with second-order MUSCL
  (no limiter), Green-Gauss gradients, implicit Euler pseudo-time with CFL 50,
  BDF2 dual time stepping at \(\Delta t = 0.025\,c/U_\infty\), 25 inner
  iterations per step (about 3 orders of residual drop). Convective units:
  \(c = U_\infty = \rho = 1\), \(\mu = 1/Re\).
* Each angle runs from a uniform free stream to \(t = 60\); `u, v, p` on the nodes
  are stored every \(0.2\) time units and \(C_l, C_d\) every step
  (`data/naca_aXX.npz`, ~90 MB per angle). Only the developed window
  \(t \ge 30\) is used (150 snapshots per angle).

Validation against Kurtulus (2015, Int. J. Micro Air Vehicles 7(3) 301-326):
see `plots/validation.md` once the sweep is complete. The previous in-house
Fourier/penalization solver was retired because on its \(512\times256\) grid the
airfoil was four cells thick and the flow shed at \(4^\circ\) with
\(St = 0.40\) where the reference is steady; its data and solver are archived in
`data/archive_512x256_coarse/`.

The modelled state stacks both velocity components on the \(N = 37\,572\) nodes,
\(d = 2N = 75\,144\), each node weighted by the square root of its dual-cell
area so that the Euclidean norm is the discrete \(L^2\) norm of the velocity
field (`problem.py`); all methods are fit and scored in that norm.

## Parametric setup: two test scenarios over the angle-of-attack range

| split         | angle(s) of attack `alpha` (deg)                | snaps/angle | columns |
|---------------|-------------------------------------------------|------------:|--------:|
| `train`       | 4 angles evenly on `[8, 20]`: `{8, 12, 16, 20}` |         150 |     600 |
| `val_interp`  | 2 held-out interior `{10, 18}`                  |         150 |     300 |
| `val_extrap`  | 1 nearer held-out `alpha = 22`                  |         150 |     150 |
| `test_interp` | 1 held-out interior `alpha = 14`                |         150 |     150 |
| `test_extrap` | 1 deep-stall `alpha = 24`                       |         150 |     150 |

## Grid-adaptive, per-variant budget selection

The Lipschitz budget `L` is **not** hard-coded: `budget_select.py` chooses it
automatically, **separately for `p=2` and `p=3`**, by minimising the
geometric-mean validation objective `sqrt(e_interp * e_extrap)` on a grid that is
progressively refined to `0.01` precision.

## Files

* `make_mesh.py`      -- body-fitted hybrid mesh generator (gmsh for the
  exterior triangles), writes `data/naca0012_lam.su2`.
* `su2_lam_config.py` -- SU2 configuration for the laminar unsteady case.
* `run_param.py`      -- angle sweep driver: runs SU2 per angle and packs the
  snapshots into `data/naca_aXX.npz` (`SU2_CFD` must be on PATH or in `$SU2_CFD`).
* `problem.py`        -- angle-of-attack splits, area weights, lifting helpers.
* `preprocess.py`     -- centering (shared with `lin_adv_diff` / `burgers2d`).
* `baselines.py`      -- POD, Greedy-QM, Greedy-CM, Geelen-QM (self-contained).
* `budget_select.py`  -- grid-adaptive, per-variant `L` selection.
* `methods.py`        -- method registry (no hard-coded `L`).
* `plots.py`          -- all figures (unstructured-mesh field panels).
* `run_experiment.py` -- resumable driver; phases
  `budget | sweep | plots | field | report | all`.
* `sv_decay.py`       -- the n-width (slow-decay) evidence.

## Running

```bash
python make_mesh.py --out data/naca0012_lam.su2 --preview data/mesh_preview.png
SU2_CFD=/path/to/SU2_CFD python run_param.py     # ~3.5 h per angle on 2 cores
python run_experiment.py budget     # select L* per variant (grid-adaptive)
python run_experiment.py sweep      # all methods, r = 1..10, at their L*
python run_experiment.py plots      # figures from the checkpoints
python run_experiment.py field      # speed-field panels (interp + extrap)
```

## Figures and report (`plots/`)

* `naca_error_vs_r.*`     -- error vs `r`: train / interpolation / extrapolation.
* `naca_certificate.*`    -- attained Lipschitz certificate vs the set budget.
* `naca_active_features.*`-- active nonlinear features vs `r`.
* `naca_budget_study.*`   -- test errors, validation objective, active-set size vs `L`.
* `naca_field_interp.*` / `naca_field_extrap.*` -- speed-field reconstruction and
  absolute-error panels on the wake at the two test angles (hardest snapshot of
  the split; annotated error is that of the whole split).
* `naca_report.json`      -- the paper-reportable numbers.
