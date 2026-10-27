# Greedy Orthogonal Polynomial Manifold (Greedy-OPM)

A nonlinear trial manifold for model reduction,

```
Gamma(shat) = s_ref + V shat + Bbar q(shat),     V^T V = I,  V^T Bbar = 0,
```

whose nonlinear features `p` are multivariate orthogonal polynomials built by
a Vandermonde-with-Arnoldi recurrence on the training coordinates, and whose
coefficient matrix `Bbar` is selected under a **certified Lipschitz budget**: the
decoder's sensitivity on a dilated neighborhood of the training data is
bounded by a user-prescribed budget `P`. 

This repository contains the experiments to reproduce the results in the paper.

## Use

```python
import jax
jax.config.update("jax_enable_x64", True)
from greedyopm import GreedyOPM

model = GreedyOPM(
    r=10,            # reduced dimension
    s=50,            # candidate pool of leading singular vectors (s > r)
    p=[2, 3],        # polynomial degrees of the nonlinear features
    L=100.0,         # Lipschitz budget (> 1); the only tuned hyperparameter
    delta=0.1,       # extrapolation margin: certify on (1 + delta) * shat
    n_reorth=4,      # Gram-Schmidt passes in the Arnoldi recurrence
    n_grad_samples=250,   # gradient-probe subsample in the greedy loop
).fit(S)             # S: (d, n) centered/normalized training snapshots

S_hat = model.encode(S)                 # (r, n)
S_rec = model.reconstruct(S)            # encode-decode
model.reconstruction_error(S)           # relative Frobenius error
model.certificate_                      # attained L_hat_Gamma(delta) <= L
model.n_active_nonlinear                # active orthogonal-polynomial features
model.set_budget(30.0)                  # closed-form recalibration at a new L
                                        # (same modes; new active set / lifting)
```

## Layout

```
src/greedyopm/
  greedy.py          GreedyOPM: greedy mode selection + budget-calibrated lifting
  greedy_fixed.py    GreedyFixedOPM: all features active (unregularized ablation)
  budget.py          calibrate_multiplier, shrink_factors, feature_gains, ...
  polynomials/       multivariate Vandermonde-with-Arnoldi (build / eval / gradients)
experiments/
  lin_adv_diff/      periodic linear advection-diffusion, parametric in the wave speed
  burgers2d/         2D viscous Burgers, parametric in the Reynolds number
  boussinesq/        2D Boussinesq rising thermal plume, parametric in the Rayleigh number
  budget_study.py, stability_study.py, plot_training_times.py, cpn_baseline.py, ...
tests/               pytest regression tests (budget calibration, Greedy-OPM)
```

Each experiment directory has its own `README.md`, a `problem.py` (data and
parametric splits), `methods.py` (the compared manifolds: POD, Greedy-QM,
Greedy-CM, Geelen-QM, Greedy-OPM p=2 / p=3), `budget_select.py` (validation
selection of `L`), `run_experiment.py` (resumable driver) and `plots.py`.

## Reproducing the experiments

```bash
pixi install
pixi run pytest tests -q
python experiments/lin_adv_diff/run_experiment.py all
```

To reproduce the full experiment results in the accompanying paper, run:

```bash
pixi install
bash experiments/run_all.sh
```

Note that we assume that the user has a working installation of [`pixi`](https://pixi.prefix.dev/latest/).

## Citation

If you use this code in your research, please cite the following paper:

```
@misc{
  title = {Greedy {{Orthogonal Polynomial Manifolds}} with {{Lipschitz Control}} for {{Dimensionality Reduction}} and {{Reduced Modeling}}},
  author = {Koike, Tomoki and Peherstorfer, Benjamin and Qian, Elizabeth},
  year = 2026,
  month = oct,
  number = {arXiv:2610.33784},
  primaryclass = {cs.LG},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2610.33784},
}
```