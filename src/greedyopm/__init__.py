"""GreedyOPM: Greedy Orthogonal Polynomial Manifold under a Lipschitz budget.

Public API
----------
* :class:`GreedyOPM`      -- greedy mode selection + exact feature selection
  by closed-form Lipschitz-budget calibration (the paper's Algorithm 3).
* :class:`GreedyFixedOPM` -- greedy mode selection with all nonlinear
  features active (unregularized ablation).

Budget machinery (:mod:`greedyopm.budget`):

* :func:`calibrate_multiplier` -- exact sorted-breakpoint calibration.
* :func:`shrink_factors`, :func:`feature_gains`, :func:`attained_budget`,
  :func:`certificate_from_budget`.

Polynomial utilities (:mod:`greedyopm.polynomials`):

* :func:`arnoldi_build`, :func:`arnoldi_eval`, :func:`arnoldi_eval_grad`
  and batched variants -- multivariate Vandermonde-with-Arnoldi.
"""

from .budget import (
    attained_budget,
    calibrate_multiplier,
    certificate_from_budget,
    feature_gains,
    shrink_factors,
)
from .greedy import GreedyOPM
from .greedy_fixed import GreedyFixedOPM
from .polynomials import (
    ArnoldiBasis,
    arnoldi_build,
    arnoldi_build_batch,
    arnoldi_eval,
    arnoldi_eval_grad,
    arnoldi_grad_batch,
    arnoldi_orthonormality_error,
)

__all__ = [
    "GreedyOPM",
    "GreedyFixedOPM",
    "ArnoldiBasis",
    "arnoldi_build",
    "arnoldi_build_batch",
    "arnoldi_eval",
    "arnoldi_eval_grad",
    "arnoldi_grad_batch",
    "arnoldi_orthonormality_error",
    "attained_budget",
    "calibrate_multiplier",
    "certificate_from_budget",
    "feature_gains",
    "shrink_factors",
]

__version__ = "0.2.0"
