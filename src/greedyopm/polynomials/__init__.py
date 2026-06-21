"""Polynomial utilities for GreedyOPM."""

from .arnoldi import (
    ArnoldiBasis,
    arnoldi_build,
    arnoldi_build_batch,
    arnoldi_eval,
    arnoldi_orthonormality_error,
)
from .classical.classical_tensor import evaluate_classical_tensor
from .extract_features import (
    extract_features_by_degree,
    extract_poly_features,
)
from .monomial import evaluate_monomials
from .multindex import (
    compute_parent_info,
    compute_parent_info_for_new,
    degree_slices_for_degrees,
    generate_multi_indices,
    graded_lex_multi_indices,
    graded_lex_multi_indices_for_degrees,
    new_multi_indices_on_coord_add,
)
from .orthogonalize import (
    check_orthonormality,
    orthogonalize_cholesky,
    orthogonalize_cholesky_refined,
    orthogonalize_hierarchical,
    orthogonalize_hybrid,
    orthogonalize_qr,
    orthogonalize_qr_refined,
)

__all__ = [
    "ArnoldiBasis",
    "arnoldi_build",
    "arnoldi_eval",
    "arnoldi_orthonormality_error",
    "check_orthonormality",
    "compute_parent_info",
    "compute_parent_info_for_new",
    "degree_slices_for_degrees",
    "evaluate_classical_tensor",
    "evaluate_monomials",
    "extract_features_by_degree",
    "extract_poly_features",
    "generate_multi_indices",
    "graded_lex_multi_indices",
    "graded_lex_multi_indices_for_degrees",
    "new_multi_indices_on_coord_add",
    "orthogonalize_cholesky",
    "orthogonalize_cholesky_refined",
    "orthogonalize_hierarchical",
    "orthogonalize_hybrid",
    "orthogonalize_qr",
    "orthogonalize_qr_refined",
]