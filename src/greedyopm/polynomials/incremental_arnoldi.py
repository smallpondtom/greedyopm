"""Incremental multivariate Arnoldi for the greedy mode-selection loop.

This is the performance-critical kernel of :class:`greedyopm.GreedyOPM`.
It implements Section 5.5 of the paper: adding a mode at iteration ``i``
grows the basis by the new multi-indices ``I_new`` (alpha with alpha_i >= 1),
and each new psi_alpha is built by the Arnoldi recurrence using the
committed old psi values verbatim.

Key design choices
------------------
1. **Vectorise across candidate modes.**  At iteration ``i`` the ``s - i + 1``
   candidate modes j' differ only in the new coordinate's value vector;
   the old psi rows and the structure of ``I_new`` are shared.  The per-
   candidate incremental step is therefore jax-vmapped.

2. **Precompute parent positions in the combined buffer.**  Every new
   multi-index alpha in ``I_new`` has parent beta = alpha - e_i.  Its row
   lives either in the committed Psi_old (if alpha_i == 1) or earlier in
   Psi_new (if alpha_i >= 2).  We store a single int32 index into the
   concatenated (Psi_old || Psi_new) buffer; this index is the same across
   all candidates.

3. **Graded-lex order of ``I_new`` guarantees parents come first.**  For
   any alpha in ``I_new``, its parent either (a) has alpha_i - 1 >= 0 and
   therefore has strictly smaller total degree when alpha_i >= 2 (so is
   earlier in grlex within ``I_new``), or (b) has alpha_i = 0, hence lives
   in Psi_old.  Either way the parent is available when we process alpha.

4. **Compressed scoring in the SVD-reduced space.**  All feature-norm
   computations use the (s,)-sized projections  w_P = W_s^T Psi^T  rather
   than the full n-sized Psi, which keeps per-iteration memory at O(s * M)
   plus the shared O(M * n) Psi_old.

Notation
--------
* M_old   : number of committed rows (= 1 + i + m_{i-1} at iteration i)
* m_new   : number of new rows for I_new (includes new linear + new NL)
* M_new   : M_old + m_new
* n_cand  : number of candidate modes s - i + 1
* s       : candidate pool size
* n       : number of snapshots
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# State container for the committed basis at iteration i.
# ---------------------------------------------------------------------------


class IncrementalArnoldiState(NamedTuple):
    """Committed Arnoldi state at the end of iteration ``i``.

    Attributes
    ----------
    Psi : (M, n)
        Orthonormal feature matrix in grlex order.  Includes degree-0,
        degree-1 (for the ``i`` committed modes), and nonlinear features.
    H : (M, M)
        Upper-triangular recurrence matrix in grlex order.
    ell_arr : (M,) int32
        Parent coordinate index for each row.  -1 for the degree-0 row.
    parent_arr : (M,) int32
        Row index of the parent of each row.  -1 for the degree-0 row.
    w_P : (s, M)
        Compressed inner products  W_s^T Psi^T.  Maintained incrementally.
    """
    Psi: jnp.ndarray
    H: jnp.ndarray
    ell_arr: jnp.ndarray
    parent_arr: jnp.ndarray
    w_P: jnp.ndarray


def initial_state(n: int, s: int, M_max: int,
                  W_s: jnp.ndarray) -> IncrementalArnoldiState:
    """Initial state:  Psi = [1_n^T], H = [[1]].

    ``M_max`` is the maximum number of rows Psi can grow to over the full
    ``r`` iterations; we do NOT pre-allocate to M_max here because the
    shapes change per iteration (JAX prefers static shapes per call, but
    across iterations we rebuild the state).

    The compressed projection  w_P[:, 0] = W_s^T * 1_n  is computed and
    cached so subsequent updates don't need to form it again.
    """
    dtype = W_s.dtype
    Psi = jnp.ones((1, n), dtype=dtype)
    H = jnp.ones((1, 1), dtype=dtype)
    ell_arr = jnp.array([-1], dtype=jnp.int32)
    parent_arr = jnp.array([-1], dtype=jnp.int32)
    # w_P[:, 0] = W_s^T * 1_n  = sum of columns of W_s  (for the constant row)
    w_P = (W_s.T @ Psi.T)       # (s, 1)
    return IncrementalArnoldiState(Psi, H, ell_arr, parent_arr, w_P)


# ---------------------------------------------------------------------------
# Core incremental build kernel  (single candidate).
# ---------------------------------------------------------------------------
#
# Given the committed (Psi_old, H_old) and a new coordinate vector c_new
# of shape (n,), build the new rows Psi_new of shape (m_new, n) by
# applying the Arnoldi recurrence with X_new = diag(c_new):
#
#   v = c_new * Psi_combined[parent_idx[k]]
#   h = (1/n) Psi_combined @ v   (masked to previous rows)
#   v -= h @ Psi_combined
#   (CGS twice:  repeat the previous two lines)
#   H_col_padded = h (over (M_old + m_new))
#   Psi_new[k] = v / ||v||
#
# After processing all k new rows in grlex order within I_new, the combined
# Psi is (Psi_old || Psi_new) with m_new new rows.
#
# The key trick is that `Psi_combined` is formed as a logical (M_total, n)
# block where rows beyond the current k are zero; the mask (idx < M_old + k)
# then enforces the "only project against previously built" rule.
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(4,))
def _incremental_arnoldi_one_candidate(
    Psi_old: jnp.ndarray,               # (M_old, n)
    c_new: jnp.ndarray,                 # (n,)  coordinate values for X_new
    parent_idx: jnp.ndarray,            # (m_new,) int32 -- into (Psi_old||Psi_new)
    new_coord_idx: int,                 # jnp.int32 scalar (not used directly)
    n_reorth: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build the new rows of Psi and the cross/diagonal blocks of H.

    Returns
    -------
    Psi_new : (m_new, n)
    H_cross : (M_old, m_new)  -- old-to-new H block
    H_new   : (m_new, m_new)  -- upper-triangular new block
    """
    M_old, n = Psi_old.shape
    m_new = parent_idx.shape[0]
    M_total = M_old + m_new

    dtype = Psi_old.dtype
    inv_n = jnp.array(1.0 / n, dtype=dtype)

    # Combined buffer: rows [0:M_old] are Psi_old (frozen), rows [M_old:] are
    # built on-the-fly.  We pre-fill the top block with Psi_old and leave
    # zeros in the new block.
    Psi_combined = jnp.zeros((M_total, n), dtype=dtype)
    Psi_combined = Psi_combined.at[:M_old].set(Psi_old)

    H_cross = jnp.zeros((M_old, m_new), dtype=dtype)
    H_new = jnp.zeros((m_new, m_new), dtype=dtype)

    idx_total = jnp.arange(M_total)

    def body(k, state):
        Psi_buf, H_cross, H_new = state
        # Parent value lookup from the combined buffer (position already built).
        parent_val = Psi_buf[parent_idx[k]]
        v = c_new * parent_val                                      # X_new @ parent

        # Only rows with row-index < M_old + k have been built yet.
        row_k_total = M_old + k
        mask = (idx_total < row_k_total).astype(dtype)

        def one_pass(carry, _):
            v, h_col = carry
            h = inv_n * (Psi_buf @ v) * mask                        # (M_total,)
            v_new = v - h @ Psi_buf
            h_col = h_col + h
            return (v_new, h_col), None

        (v, h_col), _ = jax.lax.scan(
            one_pass, (v, jnp.zeros(M_total, dtype=dtype)),
            None, length=n_reorth + 1,
        )

        nrm = jnp.sqrt(jnp.maximum(inv_n * jnp.sum(v * v), 1e-300))

        # Commit the new row to the buffer.
        Psi_buf = Psi_buf.at[row_k_total].set(v / nrm)

        # Split h_col into the cross (old) and new blocks.
        h_old = h_col[:M_old]                                        # (M_old,)
        h_new = h_col[M_old:]                                        # (m_new,)
        # The diagonal entry H_new[k, k] is the normalisation constant.
        H_cross = H_cross.at[:, k].set(h_old)
        H_new = H_new.at[:, k].set(h_new)
        H_new = H_new.at[k, k].set(nrm)

        return Psi_buf, H_cross, H_new

    Psi_combined, H_cross, H_new = jax.lax.fori_loop(
        0, m_new, body, (Psi_combined, H_cross, H_new),
    )

    Psi_new = Psi_combined[M_old:]
    return Psi_new, H_cross, H_new


# ---------------------------------------------------------------------------
# Vectorised call across candidates.
# ---------------------------------------------------------------------------


def incremental_arnoldi_vmap(
    Psi_old: jnp.ndarray,         # (M_old, n) -- shared
    c_new_all: jnp.ndarray,       # (n_cand, n)
    parent_idx: jnp.ndarray,      # (m_new,)  -- shared
    *,
    new_coord_idx: int,
    n_reorth: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """vmap of :func:`_incremental_arnoldi_one_candidate` over candidates.

    Parameters
    ----------
    Psi_old : (M_old, n)
    c_new_all : (n_cand, n)
        One coordinate value-vector per candidate.
    parent_idx : (m_new,)  int32
        Parent row in the combined buffer (shared across candidates).
    new_coord_idx : int
        0-based index ``i`` of the newly added coordinate (used for
        out-of-sample evaluation; stored into the basis state only on
        commit).  Purely for bookkeeping here.
    n_reorth : int

    Returns
    -------
    Psi_new_all : (n_cand, m_new, n)
    H_cross_all : (n_cand, M_old, m_new)
    H_new_all   : (n_cand, m_new, m_new)
    """
    fn = partial(
        _incremental_arnoldi_one_candidate,
        new_coord_idx=new_coord_idx,
        n_reorth=n_reorth,
    )
    return jax.vmap(
        fn, in_axes=(None, 0, None), out_axes=(0, 0, 0),
    )(Psi_old, c_new_all, parent_idx)


# ---------------------------------------------------------------------------
# Compression of new Psi into SVD-reduced space  (done once per candidate;
# tiny relative to the build).  Vectorised across candidates.
# ---------------------------------------------------------------------------


@jax.jit
def compress_psi_batch(
    W_s: jnp.ndarray,              # (n, s)
    Psi_new_all: jnp.ndarray,       # (n_cand, m_new, n)
) -> jnp.ndarray:
    """w_P_new = W_s^T Psi_new^T  for each candidate.

    Returns shape ``(n_cand, s, m_new)``.
    """
    return jnp.einsum("ns,cmn->csm", W_s, Psi_new_all)


# ---------------------------------------------------------------------------
# Commit helper: extend the committed state using the WINNING candidate's
# blocks.  All operations are O(M) or O(M^2) in Python; no big re-computation.
# ---------------------------------------------------------------------------


def commit_winner(
    state: IncrementalArnoldiState,
    Psi_new: jnp.ndarray,            # (m_new, n)
    H_cross: jnp.ndarray,            # (M_old, m_new)
    H_new: jnp.ndarray,              # (m_new, m_new)
    w_P_new: jnp.ndarray,            # (s, m_new)
    new_ell_arr: np.ndarray,         # (m_new,)  int
    new_parent_arr: np.ndarray,      # (m_new,)  int  -- row index in the
                                     #                   COMMITTED (new) Psi
) -> IncrementalArnoldiState:
    """Extend the committed state with the winner's new block."""
    M_old = state.Psi.shape[0]
    m_new = Psi_new.shape[0]
    M_new = M_old + m_new

    # -- Psi  ------------------------------------------------------------
    Psi = jnp.concatenate([state.Psi, Psi_new], axis=0)               # (M_new, n)

    # -- H  (upper-triangular grlex)  ------------------------------------
    #   H_new_full = [[H_old ,  H_cross ],
    #                 [0     ,  H_new   ]]
    H_top = jnp.concatenate([state.H, H_cross], axis=1)               # (M_old, M_new)
    H_bot = jnp.concatenate(
        [jnp.zeros((m_new, M_old), dtype=state.H.dtype), H_new], axis=1,
    )
    H = jnp.concatenate([H_top, H_bot], axis=0)                       # (M_new, M_new)

    # -- ell_arr, parent_arr  --------------------------------------------
    ell_arr = jnp.concatenate(
        [state.ell_arr, jnp.asarray(new_ell_arr, dtype=jnp.int32)]
    )
    parent_arr = jnp.concatenate(
        [state.parent_arr, jnp.asarray(new_parent_arr, dtype=jnp.int32)]
    )

    # -- w_P  (incremental update: just append columns) -------------------
    w_P = jnp.concatenate([state.w_P, w_P_new], axis=1)               # (s, M_new)

    return IncrementalArnoldiState(Psi, H, ell_arr, parent_arr, w_P)