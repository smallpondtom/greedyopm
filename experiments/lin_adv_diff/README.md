# Linear advection-diffusion experiment (interpolation + extrapolation)

Comparison of the budget-calibrated Greedy-OPM against POD, the plain greedy
quadratic/cubic manifolds, the POD-encoder quadratic manifold and CPN-S on
the periodic linear advection-diffusion equation

    u_t + c u_x = nu u_xx ,   x in [0, 1) ,   c = 1 ,   nu = 1e-4 ,

with a Gaussian pulse initial condition, solved exactly in Fourier space on
`d = 1024` grid points.  The tiny viscosity makes the problem
advection-dominant (slow singular-value decay, transport n-width barrier).

## Parametric setup: the initial pulse width

The varied parameter is the **initial pulse width `w`** (squared width of
the Gaussian, `u0 = exp(-(x - x0)^2 / w) / sqrt(pi w)`); the wave speed is
fixed at `c = 1` so every trajectory traverses the domain once over the
window `T = 1`.  Diffusion widens a pulse by only `4 nu T = 4e-4` over the
window, so widths outside the training range by more than that are states
no training trajectory contains; interior widths are genuine interpolation.
Narrower pulses carry more high-frequency content and need more modes, so
**extrapolation targets a narrower pulse** than any training one.

| split         | width(s) `w`                                   | snaps/width | columns |
|---------------|-------------------------------------------------|------------:|--------:|
| `train`       | 4 widths log-spaced on `[1e-3, 4e-3]`           |         120 |     480 |
| `val_interp`  | 2 interior geometric midpoints `{1.26e-3, 3.17e-3}` |     100 |     200 |
| `val_extrap`  | `0.8 e-3` (just below the range)                |         100 |     100 |
| `test_interp` | 1 random interior width `w = 2.81e-3`           |         300 |     300 |
| `test_extrap` | `0.5 e-3` (half the narrowest training pulse)   |         300 |     300 |

All statistics come from the training split only; every error is a physical
relative Frobenius error.  (The previous version of this example varied the
wave speed over `[1, 10]`; because a speed only re-indexes the same set of
states on a periodic domain, its interpolation and extrapolation errors were
identical.  That version, with its code, figures and JSONs, is archived in
`__old__/speed_param_20261009/`.)

## Grid-adaptive, per-variant budget selection

The only Greedy-OPM hyperparameter is the Lipschitz budget `P` (dilation
`delta = 0.1` fixed).  `budget_select.py` chooses it automatically,
separately for `p=2` and `p=3`, by minimising the geometric-mean validation
objective `sqrt(e_interp * e_extrap)` over a progressively refined grid.

**Selected budgets:** `P*_{p=2} = 190.89`, `P*_{p=3} = 1798.95`.

## Files

* `problem.py`        -- spectral solution and the five parametric splits.
* `preprocess.py`     -- centering normalisation.
* `baselines.py`      -- POD, Greedy-QM, Greedy-CM, Geelen-QM (self-contained).
* `../cpn_baseline.py`-- CPN-S (released code in `../third_party/CPN_MOR`).
* `budget_select.py`  -- grid-adaptive, per-variant `P` selection.
* `methods.py`        -- method registry, legend labels and styles.
* `plots.py`          -- all figures.
* `run_experiment.py` -- resumable driver; phases
  `budget | sweep | cpn | noise | plots | field | fieldplot | all`.

## Running

    python run_experiment.py budget     # select P* per variant (grid-adaptive)
    python run_experiment.py sweep      # all methods, r = 1..10, at their P*
    python run_experiment.py cpn        # CPN-S tolerance sweep (one run per n)
    python run_experiment.py noise      # additive-noise robustness at r = 10
    python run_experiment.py plots      # figures from the checkpoints
    python run_experiment.py field      # space-time panels (refits, caches images)
    python run_experiment.py fieldplot  # re-plot the panels from the cache

## Methods (internal name -> legend label)

* `POD` -> POD: leading-`r` linear projection (reference).
* `Greedy-QM` -> Plain Greedy (p=2): greedy modes + full quadratic monomial shell.
* `Greedy-CM` -> Plain Greedy (p=3): greedy modes + full cubic monomial shell.
* `Geelen-QM` -> POD-Encoder QM: leading-`r` modes + group-sparse quadratic columns.
* `Greedy-OPM (p=2)` / `(p=3)`: greedy modes + orthonormal features under the
  certified Lipschitz budget (degrees {2} and {2,3}).
* `CPN-S` -> CPN-S (p=3): Bensalah, Nouy & Soffo (2026), released code with the
  full degree-3 dictionary; `n` is an outcome of the tolerance sweep.

## Results (relative reconstruction error, `r = 10`)

| method               | training | within range (`w=2.81e-3`) | outside range (`w=0.5e-3`) |
|----------------------|---------:|---------------------------:|---------------------------:|
| POD                  | 3.9e-01  | 2.6e-01                    | 6.3e-01                    |
| POD-Encoder QM       | 1.1e-01  | 5.3e-02                    | 3.2e-01                    |
| Plain Greedy (p=2)   | 3.3e-02  | 2.5e-02                    | 1.7e-01                    |
| Greedy-OPM (p=2)     | 2.5e-02  | 2.6e-02                    | 1.4e-01                    |
| Plain Greedy (p=3)   | 1.8e-05  | 5.8e-02                    | 8.8e-01                    |
| CPN-S (p=3)          | 2.5e-03  | 2.3e-03                    | 5.4e-02                    |
| **Greedy-OPM (p=3)** | 5.2e-04  | **1.6e-03**                | **6.4e-02**                |

Takeaways:

1. Interpolation and extrapolation now differ by a factor 2.5 already for
   POD, so the extrapolation split is a genuine out-of-range test.
2. The plain cubic manifold overfits (training error 1.8e-5) and is worse
   than POD outside the range; the budgeted Greedy-OPM (p=3) keeps the cubic
   accuracy inside the range and stays an order of magnitude below POD
   outside it.
3. CPN-S is the strongest method at very small `n` and comparable to
   Greedy-OPM (p=3) at `r = 10`, with a decoder of 827 distinct basis
   functions against 159 active features.

## Figures and report (`plots/`)

* `lin_adv_diff_error_vs_r.*`        -- error vs `r`, within / outside range.
* `lin_adv_diff_active_features.*`   -- active nonlinear features vs `r`.
* `lin_adv_diff_budget_study.*`      -- validation objective and active set vs `P`.
* `lin_adv_diff_certificate.*`       -- attained certificate vs the budget `P*`.
* `lin_adv_diff_noise.*`             -- additive-noise robustness at `r = 10`.
* `lin_adv_diff_field_test_*.*`      -- space-time reconstruction / error panels
  (4 x 4, ground truth top-left); `*_field_test_*.npz` cache the images.

`lin_adv_diff_report.json` collects the paper-reportable numbers.
