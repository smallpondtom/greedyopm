# 2D NACA 0012 experiment (interpolation + extrapolation over angle of attack)

Comparison of the budget-calibrated Greedy-OPM against POD, Greedy-QM, Greedy-CM,
and Geelen-QM on a 2-D incompressible Navier-Stokes flow past a NACA 0012 airfoil
at \(Re_c = 1000\), with the **angle of attack** \(\alpha\) as the varied
parameter (the aerodynamic analogue of the Reynolds-number sweep in the 2-D
Burgers example). Across the range the wake goes from thin attached-shear-layer
shedding to progressively deeper stall, so no low-dimensional linear subspace
superposes the family: the parametric snapshot matrix has a slowly decaying
Kolmogorov n-width (see `sv_decay.py`), the regime the Greedy Orthogonal
Polynomial Manifold targets.

The modelled state stacks both velocity components on the `512 x 256` grid, so
`d = 2 * ny * nx = 262144`. Because `d` is an order of magnitude larger than the
Burgers case, every split is compressed once to a shared orthonormal basis `Q`
built from the union of all splits' developed-window snapshots (centered by the
training mean). `Q` is an isometry on that span (projection residual `~7e-12`,
`K = 683`), so the methods are fit and evaluated in the small `K`-dimensional
space while every reported number remains a true full-state velocity error. This
is a lossless implementation detail; Burgers, smaller in `d`, works in the full
space directly.

## Solver / data

`naca_solver.py` -- Fourier pseudo-spectral on a periodic box with 2/3
dealiasing, rotational-form nonlinearity, Fourier-space Leray projection, SSP-RK3
advection, exact spectral diffusion, Brinkman/volume penalization for the immersed
airfoil, a downstream fringe/sponge, and Strang splitting, in JAX. `run_param.py`
sweeps the angles; each developed window (`t >= 15`) contributes 76 snapshots.

## Parametric setup: two test scenarios over the angle-of-attack range

| split         | angle(s) of attack `alpha` (deg) | snaps/angle | columns |
|---------------|----------------------------------|------------:|--------:|
| `train`       | 4 angles evenly on `[4, 16]`: `{4, 8, 12, 16}` | 76 | 304 |
| `val_interp`  | 2 held-out interior `{6, 14}`    |          76 |     152 |
| `val_extrap`  | 1 nearer held-out `alpha = 20`   |          76 |      76 |
| `test_interp` | 1 held-out interior `alpha = 10` |          76 |      76 |
| `test_extrap` | 1 deep-stall `alpha = 24`        |          76 |      76 |

The training snapshots per angle are modest (76, just above the `p=3` feature
count of 275 at `r=10` once stacked over 4 angles), and the extrapolation test
sits well beyond the training range in the deep-stall regime -- the combination
is what makes a loose manifold unstable and the budget matter.

## Grid-adaptive, per-variant budget selection (the key figure)

The Lipschitz budget `L` is **not** hard-coded: `budget_select.py` chooses it
automatically, **separately for `p=2` and `p=3`**, by minimising the
geometric-mean validation objective `sqrt(e_interp * e_extrap)` on a grid that is
progressively refined -- both the swept range and the interval shrink each stage
-- to `0.01` (second-decimal) precision.

**Selected budgets:** `L*_{p=2} = 46.11`, `L*_{p=3} = 112.75`.

The budget-study figure shows *why* the budget matters: as `L` grows the manifold
uses more features, the interpolation error keeps improving for a while, but the
**extrapolation error turns over and blows up** past the optimum. For `p=3` the
effect is dramatic:

| Greedy-OPM (p=3), `r=10`                    | extrapolation error (`alpha=24`) | active |
|---------------------------------------------|---------------------------------:|-------:|
| loose budget `L = 1e5` (≈ unregularised)    | **1.1e+01**                      | 275    |
| selected `L* = 112.75`                      | **9.3e-01**                      | 81     |

i.e. tuning the budget on the validation splits cuts the extrapolation error by
~12x and drops the active cubic features from 275 to 81. This is the stability
the Lipschitz certificate buys.

## Files

* `problem.py`        -- angle-of-attack splits and the lossless union-basis
  compression (state `d = 262144` -> `K = 683`).
* `preprocess.py`     -- centering (shared with `lin_adv_diff` / `burgers2d`).
* `baselines.py`      -- POD, Greedy-QM, Greedy-CM, Geelen-QM (self-contained).
* `budget_select.py`  -- grid-adaptive, per-variant `L` selection.
* `methods.py`        -- method registry (no hard-coded `L`).
* `plots.py`          -- all figures (NACA-adapted journal-quality field panel).
* `run_experiment.py` -- resumable driver; phases
  `budget | sweep | plots | field | report | all`.
* `naca_solver.py`, `run_param.py`, `sv_decay.py` -- solver, angle sweep, and the
  n-width (slow-decay) evidence.

## Running

```bash
python run_param.py                 # generate the angle-of-attack data (one .npz per angle)
python run_experiment.py budget     # select L* per variant (grid-adaptive)
python run_experiment.py sweep      # all methods, r = 1..10, at their L*
python run_experiment.py plots      # figures from the checkpoints
python run_experiment.py field      # speed-field panels (interp + extrap)
```

`problem.py` builds the compressed splits under `data/interp/` on first use (peak
`~4 GB`; run this experiment on its own).

## Results (relative full-state velocity error, `r = 10`)

| method               | train    | interpolation (`alpha=10`) | extrapolation (`alpha=24`) |
|----------------------|---------:|---------------------------:|---------------------------:|
| POD                  | 2.1e-01  | 1.8e-01                    | 7.6e-01                    |
| Geelen-QM            | 1.3e-01  | 1.2e-01                    | 9.1e-01                    |
| Greedy-QM            | 9.5e-02  | 1.0e-01                    | 1.1e+00                    |
| Greedy-CM            | **9.0e-03** | 2.0e-01                 | **1.0e+01**                |
| Greedy-OPM (p=2)     | 1.1e-01  | 9.7e-02                    | 8.1e-01                    |
| **Greedy-OPM (p=3)** | 6.1e-02  | **6.6e-02**                | 9.2e-01                    |

Takeaways:

1. **Interpolation: Greedy-OPM (p=3) wins decisively** -- `6.6e-02`, ~2.7x below
   POD (`1.8e-01`) and below every baseline. `p=2`, Greedy-QM and Geelen-QM also
   beat POD.
2. **Greedy-CM overfits -- the clearest evidence for the budget.** The cubic
   monomial manifold reaches the **lowest training error of all** (`9.0e-03`, and
   plunges further with `r`), but its interpolation error *turns over* and rises
   past `r ~ 7` to `2.0e-01` -- worse than POD -- and its extrapolation error
   **explodes to `1.0e+01`**. Greedy-OPM (p=3) uses the **same cubic degree** yet,
   through orthonormal features under the certified Lipschitz budget, stays at
   `6.6e-02` / `9.2e-01`. Same features, opposite behaviour: the budget is what
   turns cubic capacity into out-of-sample stability.
3. **Extrapolation exposes instability, and the budget contains it.** Deep-stall
   `alpha = 24` is far outside the training range and hard for *every* method
   (POD `7.6e-01`); the point is that the un-budgeted monomials over-extrapolate
   catastrophically (Greedy-CM `1.0e+01`, Greedy-QM `1.1e+00`) while the budgeted
   Greedy-OPM stays **bounded and POD-level** (`p=2` `8.1e-01`, `p=3` `9.2e-01`).
   A loose-budget Greedy-OPM (p=3) would itself reach `1.1e+01` (see the budget
   study) -- the validation-tuned budget is what keeps it bounded.
4. **The certificate is valid.** `L_hat_Gamma(delta) <= L*` for every `r` and both
   variants (the ratio panel of `naca_certificate`), binding from `r ~ 5-6`.

## Figures and report (`plots/`)

All figures are journal-quality (no titles on the quantitative panels, 300-dpi PNG
+ vector PDF), sharing the grammar of the `lin_adv_diff` / `burgers2d` examples.

* `naca_error_vs_r.*`     -- error vs `r`: train / interpolation / extrapolation.
* `naca_certificate.*`    -- attained Lipschitz certificate `L_hat_Gamma(delta)`
  vs the set budget `L*` across `r`, and the ratio panel verifying
  `certificate <= budget` (primal-dual validity).
* `naca_active_features.*`-- active nonlinear features vs `r` (Greedy-OPM p=3
  degree-split into orders 2 and 3).
* `naca_budget_study.*`   -- interpolation error, extrapolation error (the
  turnover), validation objective (with each variant's `L*` marked), and
  active-set size, all vs `L`.
* `naca_field_interp.*` / `naca_field_extrap.*` -- speed-field reconstruction (top
  row) and absolute-error (bottom row) panels at the two test angles, cropped to
  the airfoil wake with the airfoil outlined. The displayed snapshot is the
  **hardest** one in the split (least captured by the linear POD subspace), and
  the annotated relative error is the error over the **entire** test split
  (identical to `error_vs_r`). Greedy-CM's over-extrapolation speckle floods the
  extrapolation wake (its error map saturates), while Greedy-OPM stays clean.

`naca_report.json` collects the paper-reportable numbers: selected `L*` per order,
budget-selection time and evaluation count, per-`r` certificate/budget ratios and
validity, per-method / per-`r` training times, the error tables, and the run
configuration.

## Why parametric-over-angle (and not a single trajectory)

A single shedding trajectory at one angle is a periodic limit cycle and is
intrinsically low-dimensional (its POD error falls fast). The *parametric family*
over angle of attack is what decays slowly -- the vortices sit at different
positions, spacings and strengths across the sweep and no linear basis superposes
them cheaply (`sv_decay.py`, `plots/sv_decay.png`). That slow n-width is the
barrier the nonlinear manifold is meant to beat, which the interpolation result
above demonstrates.
