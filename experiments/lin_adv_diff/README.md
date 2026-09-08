# Linear advection-diffusion experiment (interpolation + extrapolation)

Comparison of the budget-calibrated Greedy-OPM against POD, Greedy-QM, and
Geelen-QM on the periodic linear advection-diffusion equation

    u_t + c u_x = nu u_xx ,   x in [0, 1) ,   nu = 1e-4 ,

with a narrow Gaussian pulse, solved exactly in Fourier space on `d = 1024`
grid points.  The tiny viscosity makes the problem advection-dominant (slow
singular-value decay, transport n-width barrier).

## Parametric setup: two test scenarios over a wide speed range

The varied parameter is the **wave speed `c`**, now over a deliberately wide
range, with **two** held-out test scenarios and a validation set that
mirrors both:

| split         | wave speed(s)                        | snaps/speed | columns |
|---------------|--------------------------------------|------------:|--------:|
| `train`       | 4 speeds, evenly on `[1.0, 4.0]`     |         120 |     480 |
| `val_interp`  | 2 interior midpoints `{1.5, 3.5}`    |         100 |     200 |
| `val_extrap`  | 1 speed just past the range `c=4.45` |         100 |     100 |
| `test_interp` | 1 random interior speed `c=2.87`     |         300 |     300 |
| `test_extrap` | 1 speed beyond the range `c=4.90`    |         300 |     300 |

All statistics come from the training split only; every error is a physical
relative Frobenius error.

## Grid-adaptive, per-variant budget selection

The only Greedy-OPM hyperparameter is the Lipschitz budget `L` (dilation
`delta = 0.1` fixed).  It is **not** hard-coded anywhere: `budget_select.py`
chooses it automatically, **separately for `p=2` and `p=3`**, by minimising
the geometric-mean validation objective `sqrt(e_interp * e_extrap)` over a
grid that is progressively refined -- the swept range and interval both
shrink each stage -- down to `0.01` (second-decimal) precision.  Because the
sweep is done cheaply via `GreedyOPM.set_budget`, that precision is free; the
final r-sweep then refits fully at the selected `L*`.

**Selected budgets:** `L*_{p=2} = 1301.06`, `L*_{p=3} = 43473.26`.

For this *linear* problem the budget selection lands at a large `L`: a wave
speed only phase-shifts the solution, so the reduced coordinates stay
bounded for *any* speed (in-range or not) and the polynomial manifold
extrapolates gracefully -- there is nothing to stabilise, and the budget can
be relaxed.  (Contrast the 2D Burgers case, where extrapolation destabilises
a loose manifold and the budget must be tightened.)

## Files

* `problem.py`        -- spectral solution and the five parametric splits.
* `preprocess.py`     -- centering (+ optional row min-max) normalisation.
* `baselines.py`      -- POD, Greedy-QM, Geelen-QM (self-contained).
* `budget_select.py`  -- grid-adaptive, per-variant `L` selection.
* `methods.py`        -- method registry (no hard-coded `L`).
* `plots.py`          -- all figures.
* `run_experiment.py` -- resumable driver; phases
  `budget | sweep | plots | field | all`.

## Running

    python run_experiment.py budget   # select L* per variant (grid-adaptive)
    python run_experiment.py sweep    # all methods, r = 1..10, at their L*
    python run_experiment.py plots    # figures from the checkpoints
    python run_experiment.py field    # space-time panels (interp + extrap)

## Methods

Six methods, three monomial and three orthonormal / linear:

* `POD` -- leading-`r` linear projection (reference).
* `Greedy-QM` -- greedy modes + full **quadratic** monomial shell.
* `Greedy-CM` -- greedy modes + full **cubic** (degree-3) monomial shell
  `s_i s_j s_k` -- the monomial analog of Greedy-OPM (p=3).
* `Geelen-QM` -- leading-`r` modes + group-sparse quadratic columns.
* `Greedy-OPM (p=2)` / `(p=3)` -- greedy modes + orthonormal features under
  the certified Lipschitz budget (degrees {2} and {2,3}).

## Results (relative reconstruction error, `r = 10`)

| method            | interpolation (`c=2.87`) | extrapolation (`c=4.90`) |
|-------------------|-------------------------:|-------------------------:|
| POD               | 5.0e-01                  | 5.1e-01                  |
| Geelen-QM         | 1.6e-01                  | 1.6e-01                  |
| Greedy-QM         | 2.0e-02                  | 2.1e-02                  |
| Greedy-OPM (p=2)  | 1.0e-02                  | 1.1e-02                  |
| Greedy-CM         | 3.9e-05                  | 6.6e-05                  |
| **Greedy-OPM (p=3)** | **1.5e-05**           | **2.8e-05**              |

Takeaways:

1. **Extrapolation is as easy as interpolation here** for every method
   (`test_extrap ~ test_interp`): the wave speed does not push the reduced
   coordinates out of distribution, so nothing blows up -- a genuine feature
   of the linear transport problem.
2. **Cubic degree is decisive on this smooth problem.** Both cubic methods
   (Greedy-CM and Greedy-OPM p=3) reach ~1e-5, roughly 3 orders below the
   quadratic methods; Greedy-CM does not overfit here (bounded coordinates),
   so it tracks Greedy-OPM (p=3) to within ~2.5x. Greedy-OPM (p=3) still
   edges it out through the orthonormal conditioning of its features.
3. **POD fails** on the transport barrier (~0.5) and **Geelen-QM** trails at
   ~0.16.

(The complementary 2D-Burgers experiment is where the *budget* separates
Greedy-OPM (p=3) from Greedy-CM: there the cubic monomials overfit and the
budget prevents it.)

## Figures and report (`plots/`)

All figures are journal-quality (no titles, 300-dpi PNG + vector PDF).

* `lin_adv_diff_error_vs_r.*`        -- error vs `r`: train / interp / extrap.
* `lin_adv_diff_certificate.*`       -- the attained Lipschitz certificate
  `L_hat_Gamma(delta)` vs the set budget `L*` for every `r` and both orders;
  the ratio panel verifies `certificate <= budget` (primal-dual validity).
* `lin_adv_diff_active_features.*`   -- active nonlinear features vs `r`.
* `lin_adv_diff_budget_study.*`      -- the grid-adaptive selection:
  interpolation error, extrapolation error, validation objective (with each
  variant's `L*` marked), and active-set size, all vs `L`.
* `lin_adv_diff_field_test_interp.*` / `..._field_test_extrap.*` -- space-time
  reconstruction / error panels at the two test speeds.

`lin_adv_diff_report.json` collects the paper-reportable numbers: the
selected `L*` per order, the budget-selection time and evaluation count, the
per-`r` certificate/budget ratios and their validity, the per-method /
per-`r` training times, the error tables, and the run configuration.
