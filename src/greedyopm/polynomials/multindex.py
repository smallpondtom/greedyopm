from __future__ import annotations 

import numpy as np
from math import comb
from typing import Dict

# Multi-index generation  (pure NumPy — used at trace time, not JIT'd)
def generate_multi_indices(r: int, q: int) -> np.ndarray:
    """All multi-indices alpha in N_0^r with |alpha| = q, in graded-lex order.

    Returns shape (count, r) with count = C(r+q-1, q).
    """
    if q == 0:
        return np.zeros((1, r), dtype=np.int32)
    if r == 1:
        return np.array([[q]], dtype=np.int32)
    rows = []
    for j in range(q, -1, -1):          # grlex: largest first component first
        sub = generate_multi_indices(r - 1, q - j)
        prefix = np.full((sub.shape[0], 1), j, dtype=np.int32)
        rows.append(np.hstack([prefix, sub]))
    return np.vstack(rows)


def graded_lex_multi_indices(r: int, p: int, *, min_degree: int = 0) -> np.ndarray:
    """All multi-indices up to total degree p in graded-lex order.

    Returns shape (count, r).
    """
    blocks = [generate_multi_indices(r, q) for q in range(min_degree, p + 1)]
    return np.vstack(blocks)


def degree_slices(r: int, p: int, *, min_degree: int = 0) -> Dict[int, slice]:
    """Return {degree: slice} mapping into the stacked multi-index array."""
    offset = 0
    slices: Dict[int, slice] = {}
    for q in range(min_degree, p + 1):
        m_q = comb(r + q - 1, q)
        slices[q] = slice(offset, offset + m_q)
        offset += m_q
    return slices


def _new_nonlinear_multi_indices(n_coords: int, p: int) -> np.ndarray:
    """Multi-indices that are NEW when the n_coords-th coordinate is added.

    These are α ∈ N₀^{n_coords} with 2 ≤ |α| ≤ p  AND  α_{n_coords} ≥ 1,
    i.e. the nonlinear features that involve the newest coordinate.

    Returns shape (m_new, n_coords) or (0, n_coords) if none exist.
    """
    if n_coords < 1 or p < 2:
        return np.empty((0, max(n_coords, 1)), dtype=np.int32)

    all_nl = graded_lex_multi_indices(n_coords, p, min_degree=2)
    # Keep only rows where the LAST component is ≥ 1
    mask = all_nl[:, -1] >= 1
    new = all_nl[mask]
    if len(new) == 0:
        return np.empty((0, n_coords), dtype=np.int32)
    return new

