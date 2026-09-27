# 2D Boussinesq rising thermal (interpolation + extrapolation)

Third numerical example: a hot Gaussian blob released at rest in a
periodic box, integrated with a Fourier pseudo-spectral solver for the
incompressible Boussinesq equations in vorticity-streamfunction form

    w_t  + u w_x  + v w_y  = nu    (w_xx  + w_yy)  + theta_x ,
    th_t + u th_x + v th_y = kappa (th_xx + th_yy) ,
    u = psi_y ,  v = -psi_x ,  -(psi_xx + psi_yy) = w .

Lengths are scaled by the blob radius `R` and velocities by the buoyancy
velocity `U = sqrt(g alpha dtheta R)`, so the only parameters are
`Re = U R / nu` and `Pr = nu / kappa` (`Ra = Re^2 Pr`); `Pr = 1` is fixed
and `Re` is what the parametric experiment varies.  The modelled state
stacks `[u; v; theta]` on a `256 x 448` grid of the box `[0, 8) x [0, 14)`
(`dx = dy = 1/32`), so `d = 3 * 256 * 448 = 344064`.

## Why this problem

The blob accelerates upward and rolls up into a mushroom cap (a buoyant
vortex pair) whose lobes wind thin sheets of temperature and vorticity
into spirals while the cap keeps rising.  Two things make the reduced
coordinates hard: the cap travels ~7 radii (transport, slow n-width
decay, as in the first two experiments), and the sheets thin and the
spirals tighten as `Re` grows, so the *shape* of the solution changes
with the parameter and extrapolating in `Re` means predicting finer
structure than the training set contains.  There is no immersed body, no
pressure solve and no mesh that depends on the parameter, so the data are
spectrally accurate and fully reproducible.

## Solver (`boussinesq.py`)

* Spectral derivatives; the streamfunction is one division by `k^2`
  (`k = 0` mode zero, i.e. the mean buoyancy is balanced hydrostatically).
* Advection products formed in physical space and 2/3-dealiased, which
  makes the advective term energy-conserving to round-off.
* Both diffusion terms integrated exactly by an integrating factor inside
  a classical RK4 stage structure (IF-RK4), so `dt = 5e-3` is set by the
  advective CFL (`~0.25`) alone.
* Float64, JIT-compiled, `lax.scan` time loop.  About 20 s per Reynolds
  number on an M-series laptop for the 2000-step window `t in [0, 10]`.

## Parametric splits (`problem.py`)

Same protocol as `lin_adv_diff` and `burgers2d`:

| split         | Reynolds number(s)                  | snaps/Re | columns |
|---------------|-------------------------------------|---------:|--------:|
| `train`       | 4 numbers, evenly on `[250, 750]`   |      101 |     404 |
| `val_interp`  | 2 interior midpoints `{333, 667}`   |       81 |     162 |
| `val_extrap`  | 1 number past the range `Re=1000`   |       81 |      81 |
| `test_interp` | 1 random interior `Re=562`          |      201 |     201 |
| `test_extrap` | 1 number well beyond `Re=1250`      |      201 |     201 |

## Verification (`validate_solver.py`)

Reference runs on a grid twice as fine (`512 x 896`, `dt/2`) bound the
discretization error of the production grid over the whole window:

| Re   | max rel. error in theta | in (u, v) |
|-----:|------------------------:|----------:|
| 1000 | 8e-4                    | 1e-4      |
| 1250 | 1.9e-3                  | 2.9e-4    |
| 2000 | 9e-3                    | 2e-3      |

which is why the range stops at `Re = 1250` for the extrapolation test
(`Re = 2000` would need the finer grid, at 4x the state dimension).  The
script also checks, on every run: mean temperature conserved to round-off,
thermal variance monotone, kinetic-energy budget
`dKE/dt = <theta v> - nu <w^2>` satisfied to `~1e-9`, mirror symmetry
about `x = Lx/2` preserved to `~1e-14`, RK4 time-step convergence
(`dt` vs `dt/2`, `1e-7`), and the initial centroid acceleration against
the theta-weighted Leray projection of the buoyancy force computed
independently in numpy (`0.224`; `0.25` for the same Gaussian in an
unbounded fluid, the difference being the periodic images).  The rise
curve starts ballistically (`a t^2 / 2`) and bends toward the `t^{1/2}`
similarity law of a 2D thermal; at `t = 10` it is still in the
transition (fitted exponent `~0.8`), which is reported, not thresholded.


## Greedy-OPM comparison (`run_experiment.py`)

Same six methods, protocol and budget selection as `lin_adv_diff` and
`burgers2d`: POD, Greedy-QM, Greedy-CM (cubic monomial shell), Geelen-QM,
Greedy-OPM (p=2) and Greedy-OPM (p=3); candidate pool `s = 50`, dilation
`delta = 0.1`, `r = 1..10`, budget selected once at `r = 10` per variant by
the grid-adaptive sweep on `sqrt(e_val_interp * e_val_extrap)`.
Preprocessing is centering by the training row mean only; errors are
relative Frobenius errors of the stacked `[u; v; theta]` state.

The fits run in the lossless compressed coordinates of `problem.py`
(`K = 497` of 1049 union snapshots, projection residual `4e-7`, no `d x K`
matrix is ever formed; `lift` streams through the cached files for the
field panels).  The whole pipeline takes ~7 minutes on two CPU cores.

**Selected budgets:** `L*_{p=2} = 481.69`, `L*_{p=3} = 4183.03`.

| `r = 10`           | train   | interp. test | extrap. test (Re=1250) | active |
|--------------------|--------:|-------------:|-----------------------:|-------:|
| POD                | 1.2e-01 | 1.2e-01      | 1.7e-01                |      0 |
| Greedy-QM          | 1.0e-02 | 8.7e-03      | 4.0e-02                |     55 |
| Greedy-CM          | 7.4e-05 | 6.9e-04      | 6.5e-02 (min 3.2e-02 at r=7) | 220 |
| Geelen-QM          | 2.0e-02 | 2.0e-02      | 7.8e-02                |     55 |
| Greedy-OPM (p=2)   | 7.4e-03 | 5.7e-03      | 3.4e-02                |     55 |
| Greedy-OPM (p=3)   | 9.3e-04 | 9.5e-04      | 2.1e-02                | 123 (55 + 68) |

Greedy-CM has the lowest training error by an order of magnitude and its
extrapolation error turns over past `r = 7`; Greedy-OPM (p=3) with the same
degrees generalizes monotonically and is the best extrapolator at every
`r >= 4`.  The budget study shows why: at a loose budget (`L = 1e5`, all 275
features active) the `p=3` extrapolation error is `2.1e-01`, worse than
POD, against `2.0e-02` at `L*`, a 10x difference, while the interpolation
error also turns over (`1.7e-03` loose vs `8.9e-04` at `L*`).  The
certificate binds from `r = 7` for `p=3` and at `r = 10` for `p=2`, and
`certificate <= budget` holds for every `r`.

Figures in `plots/`: `boussinesq_error_vs_r`, `boussinesq_active_features`,
`boussinesq_budget_study`, `boussinesq_certificate`, and the temperature and
speed field panels `boussinesq_{theta,speed}_{test_interp,test_extrap}`
(hardest snapshot of each split, annotated with the split-wide error).
Journal-ready numbers are in `plots/boussinesq_report.json`.

## Files

* `boussinesq.py`       -- the solver (standalone and importable), plus
  `diagnostics`, `vorticity`, `spectrum_tail` helpers.
* `problem.py`          -- configuration, disk-cached solver wrapper, the
  five splits, field helpers.
* `generate_data.py`    -- generate/cache every split, write
  `data/manifest.json` with per-run diagnostics (resumable).
* `plot_fields.py`      -- field panels, cross-Re comparison, diagnostics
  figure, GIF animation.
* `validate_solver.py`  -- the verification above, writes
  `plots/validation.{json,png}`.
* `preprocess.py`       -- centering (shared with `lin_adv_diff`).
* `baselines.py`        -- POD, Greedy-QM, Greedy-CM, Geelen-QM.
* `budget_select.py`    -- grid-adaptive, per-variant `L` selection.
* `methods.py`          -- method registry (no hard-coded `L`).
* `plots.py`            -- comparison figures.
* `run_experiment.py`   -- resumable driver; phases
  `budget | sweep | plots | field | report | all`.

## Running

    python generate_data.py              # all splits (~5 min on a laptop, ~3 GB in data/)
    python plot_fields.py                # plots/fields_Re*.png, compare_Re.png, diagnostics.png
    python plot_fields.py --gif --re 750 # plots/anim_Re750.gif
    python validate_solver.py            # full verification incl. the fine-grid run (~5 min)
    python validate_solver.py --quick    # without the fine-grid run
    python run_experiment.py all         # budget -> sweep -> plots -> field -> report

`data/compressed/` (a few MB) is enough for every phase except `field`,
which lifts the reconstructions through the raw `data/*.npz`.
