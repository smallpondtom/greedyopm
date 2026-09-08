# 2D viscous Burgers experiment (interpolation + extrapolation)

Comparison of the budget-calibrated Greedy-OPM against POD, Greedy-QM, and
Geelen-QM on the 2D viscous Burgers equations on the periodic square
\([0, 2\pi)^2\)

    u_t + u u_x + v u_y = nu (u_xx + u_yy) ,
    v_t + u v_x + v v_y = nu (v_xx + v_yy) ,

integrated with the Fourier pseudo-spectral / RK4 solver of
`experiments/data/burgers2d.py` from a two-scale vortical initial condition
on a uniform mean flow `(c, c)`, `c = 1`.  The state stacks both velocity
components on an `N = 128` grid, so `d = 2 N^2 = 32768`.

The nonlinearity steepens the harmonics into thin internal layers and the
mean flow sweeps them across the box (slow n-width decay); the Reynolds
number `Re = 2 pi / nu` -- which sets the layer thickness -- is the parameter
that changes the *shape* of the solution, and it is what the test varies.

## Parametric setup: two test scenarios over a wide Reynolds range

| split         | Reynolds number(s)                | snaps/Re | columns |
|---------------|-----------------------------------|---------:|--------:|
| `train`       | 4 numbers, evenly on `[140, 340]` |      ~85 |     340 |
| `val_interp`  | 2 interior midpoints `{173, 307}` |      ~79 |     158 |
| `val_extrap`  | 1 number past the range `Re=430`  |       79 |      79 |
| `test_interp` | 1 random interior `Re=265`        |      131 |     131 |
| `test_extrap` | 1 number well beyond `Re=480`     |      131 |     131 |

The range is wide and the training snapshots per Reynolds number are sparse
(`~85`, just above the `p=3` feature count of 275 at `r=10`), and the
extrapolation test sits far outside the range -- the combination is what
makes a loose manifold unstable and the budget matter.  The whole range is
fully resolved on `128^2` (spectral tail `< 1e-6`).

## Grid-adaptive, per-variant budget selection (the key figure)

The Lipschitz budget `L` is **not** hard-coded: `budget_select.py` chooses it
automatically, **separately for `p=2` and `p=3`**, by minimising the
geometric-mean validation objective `sqrt(e_interp * e_extrap)` on a grid
that is progressively refined -- both the swept range and the interval shrink
each stage -- to `0.01` (second-decimal) precision.

**Selected budgets:** `L*_{p=2} = 394.50`, `L*_{p=3} = 1558.96`.

The budget-study figure shows *why* the budget matters here: as `L` grows the
manifold uses more features and the **extrapolation error turns over** --
past the optimum a looser budget over-extrapolates and blows up.  For
`p=3` the effect is dramatic:

| Greedy-OPM (p=3), `r=10` | extrapolation error (`Re=480`) | active |
|--------------------------|-------------------------------:|-------:|
| loose budget `L = 1e5` (≈ unregularised) | **3.2e-01** | 275 |
| selected `L* = 1558.96`  | **3.9e-02**                    | 131 |

i.e. tuning the budget on the validation splits cuts the extrapolation error
by ~8x, and the loose-budget manifold is worse out-of-range than *every*
baseline.  This is the stability advantage the Lipschitz certificate buys.

## Files

* `problem.py`        -- solver wrapper, caching, and the five splits.
* `preprocess.py`     -- centering (shared with `lin_adv_diff`).
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
    python run_experiment.py field    # speed-field panels (interp + extrap)

Solver output is cached under `data/` (one `.npz` per `(nu, stride)`, ~30 s
each).  Run this experiment on its own -- alongside `lin_adv_diff` it can
exceed the ~6 GB sandbox memory budget.

## Results (relative reconstruction error, `r = 10`)

The methods are POD, `Greedy-QM` (quadratic monomials), `Greedy-CM` (cubic
monomials, the monomial analog of Greedy-OPM p=3), `Geelen-QM`, and the two
budget-calibrated `Greedy-OPM` variants.

| method            | train (`r=10`) | interpolation (`Re=265`) | extrapolation (`Re=480`) |
|-------------------|---------------:|-------------------------:|-------------------------:|
| POD               | 2.0e-01        | 2.1e-01                  | 2.6e-01                  |
| Geelen-QM         | 1.1e-01        | 1.1e-01                  | 1.7e-01                  |
| Greedy-QM         | 3.4e-02        | 2.9e-02                  | 7.9e-02                  |
| Greedy-CM         | **6.9e-04**    | 3.6e-02                  | 9.7e-02                  |
| Greedy-OPM (p=2)  | 2.5e-02        | 2.0e-02                  | 6.6e-02                  |
| **Greedy-OPM (p=3)** | 5.8e-03     | **1.0e-02**              | **3.9e-02**              |

Takeaways:

1. **Greedy-CM overfits -- the clearest evidence for the budget.**  The
   cubic monomial manifold reaches the **lowest training error of all** at
   `r=10` (6.9e-04) but its test error *turns over* and rises past `r ~ 7`,
   ending at 3.6e-02 (interp) / 9.7e-02 (extrap) -- worse than the *quadratic*
   Greedy-QM.  Greedy-OPM (p=3) uses the **same cubic degree** yet, through
   the orthonormal features under the certified Lipschitz budget, generalises
   monotonically to 1.0e-02 / 3.9e-02.  Same features, opposite behaviour:
   the budget is what turns cubic capacity into out-of-sample accuracy.
2. **Greedy-OPM (p=3) wins on both tests** -- ~3x below Greedy-CM/Greedy-QM
   on interpolation and ~2x on extrapolation, ~5-7x below POD.
3. **Extrapolation exposes instability.** A loose-budget Greedy-OPM (or an
   unregularised manifold such as Greedy-CM) over-extrapolates to ~0.1-0.3;
   the validation-tuned budget keeps Greedy-OPM (p=3) at ~0.04.  **POD (0.26)
   and Geelen-QM (0.17) fail out of range** too.
4. **The budget genuinely regularises** here: sparse per-parameter snapshots
   and a far extrapolation give the budget study a clear turnover, and the
   grid-adaptive sweep selects each variant's own interior `L*`.

## Figures and report (`plots/`)

All figures are journal-quality (no titles, 300-dpi PNG + vector PDF).

* `burgers2d_error_vs_r.*`        -- error vs `r`: train / interp / extrap.
* `burgers2d_certificate.*`       -- the attained Lipschitz certificate
  `L_hat_Gamma(delta)` vs the set budget `L*` for every `r` and both orders;
  the ratio panel verifies `certificate <= budget` (primal-dual validity,
  binding from `r=7` for `p=3` and `r=10` for `p=2`).
* `burgers2d_active_features.*`   -- active nonlinear features vs `r`.
* `burgers2d_budget_study.*`      -- interpolation error, extrapolation error
  (the turnover), validation objective (with each variant's `L*` marked),
  and active-set size, all vs `L`.
* `burgers2d_field_test_interp.*` / `..._field_test_extrap.*` -- speed-field
  reconstruction / error panels at the two test Reynolds numbers.  The
  displayed snapshot is the **hardest** one in the split -- the state least
  captured by the linear POD subspace (a method-agnostic difficulty measure)
  -- rather than a fixed time index, and the annotated relative error is the
  error over the **entire** test split (identical to the `error_vs_r`
  figure).  A fixed snapshot is not representative of an over-fitting method
  such as Greedy-CM, whose per-snapshot error spans several orders of
  magnitude (median ~1e-3 but a few catastrophic columns dominate the
  split-averaged error); at the hardest snapshot the per-method ordering
  matches `error_vs_r`, and Greedy-CM's high-frequency overfitting speckle is
  visible in the error map.

`burgers2d_report.json` collects the paper-reportable numbers: the selected
`L*` per order, the budget-selection time and evaluation count, the per-`r`
certificate/budget ratios and their validity, the per-method / per-`r`
training times, the error tables, and the run configuration.
