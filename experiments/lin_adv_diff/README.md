# Linear advection-diffusion experiment

Comparison of the budget-calibrated Greedy-OPM against POD, Greedy-QM, and
Geelen-QM on the periodic linear advection-diffusion equation

    u_t + c u_x = nu u_xx ,   x in [0, 1) ,   nu = 1e-4 ,

with a narrow Gaussian pulse initial condition, solved exactly in Fourier
space on d = 1024 grid points.  The tiny nonzero viscosity makes the problem
advection-dominant (slow singular-value decay) while slowly widening and
damping the pulse, so both test regimes are genuinely out-of-distribution
yet reconstructible.

## Data splits (no validation set)

| split           | wave speed | window       | snapshots |
|-----------------|-----------:|--------------|----------:|
| `train`         | c = 1.0    | t in [0, 1)  | 1200      |
| `test_param`    | c = 1.2    | t in [0, 1)  | 600       |
| `test_forecast` | c = 1.0    | t in [1, 2)  | 600       |

Only the wave speed is varied for the parametric test; the forecast test is
contiguous future data.  The only Greedy-OPM hyperparameter is the Lipschitz
budget `L` (`CFG["L"]`, chosen from the budget study); the dilation is fixed
at `delta = 0.1`.

## Preprocessing

All methods are fit on preprocessed snapshots: the raw data is centered by
the row-wise training mean and then normalized row-wise into `[-1, 1]`
using the row-wise training minimum and maximum (`preprocess.py`).  At
reconstruction the model output is un-normalized and then un-centered, so
every reported error is measured in the original physical units.  The
statistics come from the training split only; test snapshots may leave
`[-1, 1]` slightly, which is what the extrapolation margin `delta` covers.

## Files

* `problem.py`        -- spectral solution and the three splits.
* `preprocess.py`     -- centering + row-wise min-max normalization.
* `baselines.py`      -- POD, Greedy-QM, Geelen-QM (self-contained).
* `methods.py`        -- method registry and shared configuration.
* `plots.py`          -- all figures.
* `run_experiment.py` -- resumable driver; phases
  `budget | sweep | plots | field | all`.

## Running

    python run_experiment.py budget   # error vs L at r = 8 (full refits)
    python run_experiment.py sweep    # all methods, r = 1..10
    python run_experiment.py plots    # figures from the checkpoints
    python run_experiment.py field    # space-time field panels (bonus)

Both sweeps checkpoint to JSON in `plots/` and skip finished entries, so
they can be interrupted and rerun.
