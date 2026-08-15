"""Tiny helpers for pulling features out of a ``{degree: Psi_q}`` dict."""

from __future__ import annotations

from typing import Dict, List, Tuple

import jax.numpy as jnp


def extract_poly_features(
    P_by_degree: Dict[int, jnp.ndarray],
    *,
    min_degree: int = 2,
) -> jnp.ndarray:
    """Concatenate degree >= ``min_degree`` features into a single matrix."""
    max_deg = max(P_by_degree.keys())
    blocks = [
        P_by_degree[q]
        for q in range(min_degree, max_deg + 1)
        if q in P_by_degree
    ]
    if not blocks:
        raise ValueError(f"No features of degree >= {min_degree} found")
    return jnp.concatenate(blocks, axis=0)


def extract_features_by_degree(
    P_by_degree: Dict[int, jnp.ndarray],
    *,
    min_degree: int = 2,
) -> List[Tuple[int, jnp.ndarray]]:
    """Return ordered list of ``(degree, P_q)`` pairs."""
    max_deg = max(P_by_degree.keys())
    return [
        (q, P_by_degree[q])
        for q in range(min_degree, max_deg + 1)
        if q in P_by_degree
    ]