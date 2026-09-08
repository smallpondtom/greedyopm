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

Operator Inference on the manifold (:mod:`greedyopm.opinf`):

* :func:`opinf.infer_operators`, :func:`opinf.fbct4` -- non-intrusive operator
  regression and fourth-order FBCT derivative estimation.
* :func:`opinf.simulate` / :func:`opinf.simulate_gmg` -- plain latent and
  geometric manifold-Galerkin reduced-model rollouts (the working note
  ``notes/opinf_opm.tex``).
* :func:`opinf.interp_operator_dict` / :func:`opinf.extrap_operator_dict` --
  entrywise spline interpolation / affine extrapolation of parametric operators.
"""

from .budget import (
    attained_budget,
    calibrate_multiplier,
    certificate_from_budget,
    feature_gains,
    shrink_factors,
)
from . import opinf
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
    "opinf",
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
