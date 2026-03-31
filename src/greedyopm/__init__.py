"""
    GreedyOPM: Greedy Orthogonal Polynomial Manifold for dimensionality reduction
"""

from .polynomials.monomial import evaluate_monomials
from .polynomials.classical_tensor import evaluate_classical_tensor
from .polynomials.orthogonalize import orthogonalize_hierarchical, orthogonalize_hybrid
from .polynomials.extract_features import extract_poly_features, extract_features_by_degree
from .greedy import GreedyOPM

__all__ = [
    "GreedyOPM",
    "evaluate_classical_tensor",
    "orthogonalize_hierarchical",
    "orthogonalize_hybrid",
    "extract_poly_features",
    "extract_features_by_degree",
    "evaluate_monomials",
]

__version__ = "0.1.0"