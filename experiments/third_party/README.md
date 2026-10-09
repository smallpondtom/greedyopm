# Third-party baselines

## `CPN_MOR/` -- CPN-S (Bensalah, Nouy & Soffo 2026)

Verbatim copy (without `.git`, `.idea`, `__pycache__`) of
https://github.com/JoelSOFFO/CPN_MOR at commit
`74e2228b2c64dc601fb32dff353aefbf264f169d` (2026-07-09), the released
implementation of *Nonlinear manifold approximation using compositional
polynomial networks* (SIAM J. Sci. Comput., 2026; arXiv:2502.05088).
Licence: see `CPN_MOR/LICENSE.md`.

Nothing in this directory is modified.  The comparison wrapper is
`experiments/cpn_baseline.py`; it imports `CPN_S`, `Encoder` and `Decoder`
from here and contains a verbatim copy of `CPN_S.find_n` with (i) an
`n_max` cap that abandons a run once the adaptive rule promotes more than
`n_max` modes, (ii) a switch for the polynomial index set (`index_set`:
the released default is the hyperbolic cross `prod(nu_k + 1) <= p + 1`,
which has no mixed cubic terms; the experiments use `"total_degree"`, the
full `|nu|_1 <= p` set that their code keeps as a commented-out line, so
that CPN-S has the same degree-3 dictionary as the other cubic methods) and
(iii) the console prints removed.  The equivalence of the
copy with the original (same promoted modes, same sparse coefficients, same
Lipschitz constants) was checked on the advection-diffusion data.

Dependencies (added to `pyproject.toml`): `tensap>=1.8` (PyPI), `joblib`,
`PyYAML`, `networkx`, `scikit-learn`; run `pixi install` once to update the
lock file.

Settings used in the experiments (`CPN_CFG` in each `methods.py`): the
released defaults of `configs/cpn.yaml` -- `approximation_type: sparse`
(CPN-S), degree `p = 3`, `alpha = 1`, `beta = 0.7071`, `L = 100`
(inactive: the Lipschitz check is commented out in the released `find_n`),
`n_min = 1`, `train_val_set = 1.0` (no internal validation split; the
acceptance test is evaluated on the training fit), `setting: mean_squared`,
with the total-degree index set instead of the hyperbolic cross (see above).
The background space `X_N` is spanned by the leading `N = s_pool = 50` POD
modes of the centered training snapshots (the candidate pool of the greedy
methods) instead of the tolerance-based truncation of the released driver.
The manifold dimension `n` is an outcome of the target tolerance, so each
driver sweeps the tolerance (`run_experiment.py cpn`) and stores, per
achieved `n <= R_MAX`, the run with the tightest tolerance as `CPN-S@r=n`;
every run of the sweep is kept under `_cpn_sweep` in the results JSON.
