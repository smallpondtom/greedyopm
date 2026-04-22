"""GreedyOPM: Greedy Orthogonal Polynomial Manifold for dimensionality reduction.

Public API
----------
* :class:`GreedyOPM`      -- joint greedy mode + feature selection (Algorithm 1).
* :class:`GreedyFixedOPM` -- greedy mode selection with all NL features active.

Polynomial utilities (also re-exported):

* :func:`arnoldi_build`, :func:`arnoldi_eval` -- multivariate Arnoldi.
* :func:`evaluate_monomials`, :func:`evaluate_classical_tensor`.
* :func:`orthogonalize_qr`, :func:`orthogonalize_cholesky`, and refined variants.
"""

from .greedy import GreedyFixedOPM, GreedyOPM
from .polynomials import (
    ArnoldiBasis,
    arnoldi_build,
    arnoldi_eval,
    arnoldi_orthonormality_error,
    evaluate_classical_tensor,
    evaluate_monomials,
    extract_features_by_degree,
    extract_poly_features,
    orthogonalize_cholesky,
    orthogonalize_cholesky_refined,
    orthogonalize_hierarchical,
    orthogonalize_hybrid,
    orthogonalize_qr,
    orthogonalize_qr_refined,
)

__all__ = [
    "GreedyOPM",
    "GreedyFixedOPM",
    "ArnoldiBasis",
    "arnoldi_build",
    "arnoldi_eval",
    "arnoldi_orthonormality_error",
    "evaluate_classical_tensor",
    "evaluate_monomials",
    "extract_features_by_degree",
    "extract_poly_features",
    "orthogonalize_cholesky",
    "orthogonalize_cholesky_refined",
    "orthogonalize_hierarchical",
    "orthogonalize_hybrid",
    "orthogonalize_qr",
    "orthogonalize_qr_refined",
]

__version__ = "0.1.0"