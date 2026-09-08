"""Polynomial utilities for GreedyOPM: grlex multi-indices and the
multivariate Vandermonde-with-Arnoldi orthogonalization."""

from .arnoldi import (
    ArnoldiBasis,
    arnoldi_build,
    arnoldi_build_batch,
    arnoldi_eval,
    arnoldi_eval_grad,
    arnoldi_grad_batch,
    arnoldi_orthonormality_error,
)
from .multindex import (
    compute_parent_info,
    degree_slices_for_degrees,
    generate_multi_indices,
    graded_lex_multi_indices_for_degrees,
    normalize_degrees,
)

__all__ = [
    "ArnoldiBasis",
    "arnoldi_build",
    "arnoldi_build_batch",
    "arnoldi_eval",
    "arnoldi_eval_grad",
    "arnoldi_grad_batch",
    "arnoldi_orthonormality_error",
    "compute_parent_info",
    "degree_slices_for_degrees",
    "generate_multi_indices",
    "graded_lex_multi_indices_for_degrees",
    "normalize_degrees",
]
