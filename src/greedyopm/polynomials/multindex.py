"""Multi-index utilities in graded-lexicographic (grlex) order.

All routines here are pure NumPy (invoked at trace time, not under JIT).

Layout convention
-----------------
Multi-indices are stacked by degree:  degree-0 first (the zero index), then
degree-1 (the ``r`` standard basis vectors), then each nonlinear degree q >= 2
in grlex order within the shell. This is the ordering expected by the
Arnoldi / QR / Cholesky orthogonalizers.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Degree normalisation
# ---------------------------------------------------------------------------

def _normalize_degrees(p: Union[int, Sequence[int]]) -> List[int]:
    """Coerce ``p`` into a sorted list of nonlinear degrees (>= 2)."""
    if isinstance(p, (int, np.integer)):
        return list(range(2, int(p) + 1))
    degrees = sorted(int(d) for d in p)
    if any(d < 2 for d in degrees):
        raise ValueError(
            f"All polynomial degrees must be >= 2, got {degrees}"
        )
    return degrees


# ---------------------------------------------------------------------------
# Multi-index generation (cached for repeated calls across iterations)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def generate_multi_indices(r: int, q: int) -> np.ndarray:
    """All alpha in N_0^r with |alpha| = q, in grlex order. Shape (C(r+q-1,q), r)."""
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


def graded_lex_multi_indices(
    r: int, p: int, *, min_degree: int = 0,
) -> np.ndarray:
    """All alpha with min_degree <= |alpha| <= p, stacked in grlex by degree."""
    return np.vstack([
        generate_multi_indices(r, q) for q in range(min_degree, p + 1)
    ])


def graded_lex_multi_indices_for_degrees(
    r: int,
    degrees: Sequence[int],
    *,
    include_linear: bool = True,
) -> np.ndarray:
    """Stacked multi-indices for degree 0, (optionally) 1, and ``degrees``.

    ``degrees`` is a list of nonlinear degrees (each >= 2).  Layout is
    degree-0 (single row of zeros), then degree-1 (``r`` rows) if
    ``include_linear``, then each q in sorted ``degrees`` as a grlex block.
    """
    blocks = [generate_multi_indices(r, 0)]
    if include_linear:
        blocks.append(generate_multi_indices(r, 1))
    for q in sorted(degrees):
        if q >= 2:
            blocks.append(generate_multi_indices(r, q))
    return np.vstack(blocks)


def degree_slices_for_degrees(
    r: int,
    degrees: Sequence[int],
    *,
    include_linear: bool = True,
) -> Dict[int, slice]:
    """Return ``{degree: slice}`` into the stacked multi-index array."""
    offset = 0
    slices: Dict[int, slice] = {}
    slices[0] = slice(offset, offset + 1)
    offset += 1
    if include_linear:
        slices[1] = slice(offset, offset + r)
        offset += r
    for q in sorted(degrees):
        if q >= 2:
            m_q = comb(r + q - 1, q)
            slices[q] = slice(offset, offset + m_q)
            offset += m_q
    return slices


# ---------------------------------------------------------------------------
# Incremental helpers (which multi-indices are NEW when a coordinate is added)
# ---------------------------------------------------------------------------

def new_multi_indices_on_coord_add(
    n_coords: int,
    degrees: Sequence[int],
    *,
    include_new_linear: bool = True,
) -> np.ndarray:
    """Multi-indices that appear for the first time when adding coordinate ``n_coords``.

    These are the multi-indices alpha in N_0^{n_coords} with alpha_{n_coords} >= 1
    and |alpha| in {1} ∪ degrees (when ``include_new_linear=True``).

    The returned array is already in grlex order within each degree shell,
    ordered by ascending degree.  Its LAST coordinate is >= 1 by construction.

    Arguments
    ---------
    n_coords : int
        Number of coordinates after the addition (>= 1).
    degrees : list of int
        Nonlinear degrees to include.
    include_new_linear : bool
        Whether to include the new degree-1 multi-index alpha = e_{n_coords}.

    Returns
    -------
    new : (m_new, n_coords)  int32
    """
    if n_coords < 1:
        return np.empty((0, max(n_coords, 1)), dtype=np.int32)

    blocks: list[np.ndarray] = []
    if include_new_linear:
        # alpha = e_{n_coords}
        new_linear = np.zeros((1, n_coords), dtype=np.int32)
        new_linear[0, -1] = 1
        blocks.append(new_linear)

    for q in sorted(degrees):
        if q < 2:
            continue
        all_q = generate_multi_indices(n_coords, q)
        mask = all_q[:, -1] >= 1
        if np.any(mask):
            blocks.append(all_q[mask])

    if not blocks:
        return np.empty((0, n_coords), dtype=np.int32)
    return np.vstack(blocks)


# ---------------------------------------------------------------------------
# Parent lookup
# ---------------------------------------------------------------------------

def compute_parent_info(
    multi_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each alpha in ``multi_indices``, return (ell, parent_row).

    Uses the "last non-zero coordinate" rule:  ell(alpha) = max{l : alpha_l >= 1}
    and beta(alpha) = alpha - e_ell.  For the zero index, returns (-1, -1).

    Arguments
    ---------
    multi_indices : (M, r) int

    Returns
    -------
    ell_arr : (M,) int32
    parent_arr : (M,) int32
    """
    M, r = multi_indices.shape
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


def compute_parent_info_for_new(
    old_multi_indices_padded: np.ndarray,
    new_multi_indices: np.ndarray,
    *,
    new_coord: int,
) -> np.ndarray:
    """Parent-of-each-new-alpha lookup in the (old || new) stacked buffer.

    For the incremental Arnoldi step at iteration i, every new multi-index
    alpha has parent beta = alpha - e_{new_coord} where ``new_coord`` is the
    newly added coordinate index (``i`` in 0-based indexing).  The parent
    lives either

        * in the OLD buffer (if alpha_{new_coord} == 1), or
        * earlier within the NEW buffer (if alpha_{new_coord} >= 2).

    This function returns the row index of beta in the combined buffer
    [old_rows (M_old rows); new_rows (m_new rows)].

    Arguments
    ---------
    old_multi_indices_padded : (M_old, new_coord+1)  int
        The committed multi-indices from iteration i-1, already embedded in
        the (new_coord+1)-dimensional space by appending a zero column.
    new_multi_indices : (m_new, new_coord+1)  int
    new_coord : int
        0-based index of the newly added coordinate.

    Returns
    -------
    parent_idx : (m_new,) int32
        Row index into [old_rows; new_rows].  Always in [0, M_old + m_new).
    """
    M_old = old_multi_indices_padded.shape[0]
    m_new = new_multi_indices.shape[0]

    # Hash table over the combined buffer.  Both blocks are already in
    # (new_coord+1)-dimensional space, so keys have the same length.
    index_map: dict[Tuple[int, ...], int] = {}
    for i, mi in enumerate(old_multi_indices_padded):
        index_map[tuple(mi.tolist())] = i
    for i, mi in enumerate(new_multi_indices):
        index_map[tuple(mi.tolist())] = M_old + i

    parent_idx = np.empty(m_new, dtype=np.int32)
    for i, alpha in enumerate(new_multi_indices):
        beta = alpha.copy()
        beta[new_coord] -= 1
        parent_idx[i] = index_map[tuple(beta.tolist())]

    return parent_idx