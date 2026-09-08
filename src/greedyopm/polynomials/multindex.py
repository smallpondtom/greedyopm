"""Multi-index utilities in graded-lexicographic (grlex) order.

All routines here are pure NumPy (invoked at trace time, not under JIT).

Layout convention
-----------------
Multi-indices are stacked by degree: degree-0 first (the zero index), then
degree-1 (the ``r`` standard basis vectors), then each nonlinear degree
q >= 2 in grlex order within the shell.  This is the ordering expected by
the multivariate Arnoldi orthogonalizer.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Degree normalisation
# ---------------------------------------------------------------------------


def normalize_degrees(p: Union[int, Sequence[int]]) -> List[int]:
    """Coerce ``p`` into a sorted list of nonlinear degrees (each >= 2).

    An int ``p`` means all degrees 2, ..., p; a sequence is used verbatim.
    """
    if isinstance(p, (int, np.integer)):
        return list(range(2, int(p) + 1))
    degrees = sorted(int(d) for d in p)
    if any(d < 2 for d in degrees):
        raise ValueError(f"All polynomial degrees must be >= 2, got {degrees}")
    return degrees


# ---------------------------------------------------------------------------
# Multi-index generation (cached for repeated calls across iterations)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def generate_multi_indices(r: int, q: int) -> np.ndarray:
    """All alpha in N_0^r with |alpha| = q, in grlex order.

    Arguments
    ---------
    r : int
        Number of coordinates.
    q : int
        Total degree of the shell.

    Returns
    -------
    (C(r+q-1, q), r) int32 array.
    """
    if q == 0:
        return np.zeros((1, r), dtype=np.int32)
    if r == 1:
        return np.array([[q]], dtype=np.int32)
    blocks = []
    # grlex: largest first component first
    for j in range(q, -1, -1):
        sub = generate_multi_indices(r - 1, q - j)
        prefix = np.full((sub.shape[0], 1), j, dtype=np.int32)
        blocks.append(np.hstack([prefix, sub]))
    return np.vstack(blocks)


def graded_lex_multi_indices_for_degrees(
    r: int,
    degrees: Sequence[int],
) -> np.ndarray:
    """Stacked multi-indices for degree 0, degree 1, and ``degrees``.

    ``degrees`` is a list of nonlinear degrees (each >= 2).  Layout is
    degree-0 (single row of zeros), then degree-1 (``r`` rows), then each
    q in sorted ``degrees`` as a grlex block.

    Arguments
    ---------
    r : int
    degrees : list of int

    Returns
    -------
    (M, r) int32 array with the zero multi-index first.
    """
    blocks = [generate_multi_indices(r, 0), generate_multi_indices(r, 1)]
    for q in sorted(degrees):
        if q >= 2:
            blocks.append(generate_multi_indices(r, q))
    return np.vstack(blocks)


def degree_slices_for_degrees(
    r: int,
    degrees: Sequence[int],
) -> Dict[int, slice]:
    """Return ``{degree: slice}`` into the stacked multi-index array."""
    slices: Dict[int, slice] = {0: slice(0, 1)}
    offset = 1
    slices[1] = slice(offset, offset + r)
    offset += r
    for q in sorted(degrees):
        if q >= 2:
            m_q = comb(r + q - 1, q)
            slices[q] = slice(offset, offset + m_q)
            offset += m_q
    return slices


# ---------------------------------------------------------------------------
# Parent lookup (fixed selection rule of the multivariate Arnoldi recurrence)
# ---------------------------------------------------------------------------


def compute_parent_info(
    multi_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each alpha in ``multi_indices``, return (ell, parent_row).

    Uses the last-nonzero-coordinate rule of the paper:
    ``ell(alpha) = max{l : alpha_l >= 1}`` and parent
    ``pi(alpha) = alpha - e_ell``.  In grlex order the parent always
    precedes alpha, so the Arnoldi recurrence can advance one multi-index
    at a time.  The zero index gets ``(-1, -1)``.

    Arguments
    ---------
    multi_indices : (M, r) int

    Returns
    -------
    ell_arr : (M,) int32
    parent_arr : (M,) int32
    """
    M, _ = multi_indices.shape
    ell_arr = np.full(M, -1, dtype=np.int32)
    parent_arr = np.full(M, -1, dtype=np.int32)

    index_map = {tuple(mi.tolist()): i for i, mi in enumerate(multi_indices)}

    for k in range(M):
        alpha = multi_indices[k]
        if int(alpha.sum()) == 0:
            continue
        nz = np.flatnonzero(alpha >= 1)
        ell = int(nz[-1])
        beta = alpha.copy()
        beta[ell] -= 1
        parent_arr[k] = index_map[tuple(beta.tolist())]
        ell_arr[k] = ell

    return ell_arr, parent_arr
