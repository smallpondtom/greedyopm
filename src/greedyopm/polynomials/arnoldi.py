"""Multivariate Vandermonde-with-Arnoldi orthogonalisation.

Builds empirically orthonormal polynomial features without ever forming
the ill-conditioned Vandermonde matrix.  For each alpha != 0 in graded-lex
order, a "parent pair" (ell, beta) = (last nonzero coordinate, alpha - e_ell)
is chosen, and the new polynomial is built as

    psi_alpha = (X_ell psi_beta - sum_{gamma < alpha} H_{gamma, alpha} psi_gamma)
                / H_{alpha, alpha}

where X_ell = diag(s_hat_ell) and the coefficients H_{gamma, alpha} are
stored for out-of-sample evaluation via the same recurrence.

The main advantages over QR/Cholesky on the Vandermonde:

* Phi (exponentially ill-conditioned in ``p``) is never formed.
* Evaluation at new points uses the stored H directly, with no R^{-T}
  amplification when the monomial basis is near-rank-deficient.

References
----------
Brubeck, Nakatsukasa, Trefethen, "Vandermonde with Arnoldi",
SIAM Review 63(2):405-415, 2021.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .multindex import compute_parent_info


# ---------------------------------------------------------------------------
# Stored state for out-of-sample evaluation.
# ---------------------------------------------------------------------------


class ArnoldiBasis(NamedTuple):
    """Artifacts needed to evaluate the orthogonal polynomials at new points.

    Attributes
    ----------
    H : (M, M)
        Upper-triangular recurrence matrix in grlex order.
    ell_arr : (M,) int32
        Parent coordinate index.  -1 for the degree-0 row.
    parent_arr : (M,) int32
        Row index of the parent beta(alpha) in ``multi_indices``.  -1 for
        the degree-0 row.
    multi_indices : (M, r) int32
        Multi-indices in grlex order (for reference; not used by eval).
    """
    H: jnp.ndarray
    ell_arr: jnp.ndarray
    parent_arr: jnp.ndarray
    multi_indices: jnp.ndarray


# ---------------------------------------------------------------------------
# Core build loop (JIT-compiled).  Uses classical Gram-Schmidt with "twice is
# enough" re-orthogonalisation.  The mask (idx < k) keeps the H column clean
# as Psi is still being built.
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(3,))
def _arnoldi_build_jit(
    S_hat: jnp.ndarray,
    ell_arr: jnp.ndarray,
    parent_arr: jnp.ndarray,
    n_reorth: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r, n = S_hat.shape
    M = ell_arr.shape[0]

    Psi = jnp.zeros((M, n), dtype=S_hat.dtype)
    Psi = Psi.at[0].set(jnp.ones(n, dtype=S_hat.dtype))
    H = jnp.zeros((M, M), dtype=S_hat.dtype)
    H = H.at[0, 0].set(1.0)

    idx = jnp.arange(M)
    inv_n = 1.0 / n

    def body(k, state):
        Psi, H = state
        ell = ell_arr[k]
        j = parent_arr[k]

        v = S_hat[ell] * Psi[j]              # X_ell @ psi_beta
        mask = (idx < k).astype(Psi.dtype)

        def one_pass(carry, _):
            v, H_col = carry
            h = inv_n * (Psi @ v) * mask
            v = v - h @ Psi
            H_col = H_col + h
            return (v, H_col), None

        (v, H_col), _ = jax.lax.scan(
            one_pass, (v, jnp.zeros(M, dtype=Psi.dtype)),
            None, length=n_reorth + 1,
        )

        nrm = jnp.sqrt(jnp.maximum(inv_n * jnp.sum(v * v), 1e-300))
        Psi = Psi.at[k].set(v / nrm)
        H = H.at[:, k].set(H_col)
        H = H.at[k, k].set(nrm)
        return Psi, H

    Psi, H = jax.lax.fori_loop(1, M, body, (Psi, H))
    return Psi, H


@jax.jit
def _arnoldi_eval_jit(
    S_hat_new: jnp.ndarray,
    H: jnp.ndarray,
    ell_arr: jnp.ndarray,
    parent_arr: jnp.ndarray,
) -> jnp.ndarray:
    _, M_new = S_hat_new.shape
    M = H.shape[0]
    idx = jnp.arange(M)

    W = jnp.zeros((M, M_new), dtype=S_hat_new.dtype)
    W = W.at[0].set(jnp.ones(M_new, dtype=S_hat_new.dtype))

    def body(k, W):
        ell = ell_arr[k]
        j = parent_arr[k]
        v = S_hat_new[ell] * W[j]
        mask = (idx < k).astype(W.dtype)
        v = v - (H[:, k] * mask) @ W
        W = W.at[k].set(v / H[k, k])
        return W

    W = jax.lax.fori_loop(1, M, body, W)
    return W


# ---------------------------------------------------------------------------
# User-facing API
# ---------------------------------------------------------------------------


def arnoldi_build(
    S_hat: jnp.ndarray,
    multi_indices: np.ndarray,
    *,
    n_reorth: int = 1,
) -> Tuple[jnp.ndarray, ArnoldiBasis]:
    """Build orthonormal polynomial features for the given multi-indices.

    Parameters
    ----------
    S_hat : (r, n)
        Training-time reduced coordinates.
    multi_indices : (M, r)  int
        Multi-indices in grlex order, starting with the zero index.
    n_reorth : int
        Number of additional CGS passes beyond the first (default 1).

    Returns
    -------
    Psi : (M, n)
        Orthonormal features:  (1/n) Psi Psi^T = I  in exact arithmetic.
    basis : ArnoldiBasis
        State needed by :func:`arnoldi_eval`.
    """
    multi_indices = np.asarray(multi_indices, dtype=np.int32)
    if multi_indices.ndim != 2 or int(multi_indices[0].sum()) != 0:
        raise ValueError(
            "multi_indices must be 2D with the zero multi-index first."
        )

    ell_np, parent_np = compute_parent_info(multi_indices)
    ell_arr = jnp.asarray(ell_np)
    parent_arr = jnp.asarray(parent_np)
    mi_j = jnp.asarray(multi_indices)

    Psi, H = _arnoldi_build_jit(S_hat, ell_arr, parent_arr, int(n_reorth))
    return Psi, ArnoldiBasis(H, ell_arr, parent_arr, mi_j)


def arnoldi_eval(S_hat_new: jnp.ndarray, basis: ArnoldiBasis) -> jnp.ndarray:
    """Evaluate the stored orthogonal polynomials at new coordinates."""
    return _arnoldi_eval_jit(
        S_hat_new, basis.H, basis.ell_arr, basis.parent_arr,
    )


def arnoldi_build_batch(
    S_hat_batch: jnp.ndarray,
    multi_indices: np.ndarray,
    *,
    n_reorth: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Vectorised :func:`arnoldi_build` over a batch of reduced-coordinate tensors.

    Parameters
    ----------
    S_hat_batch : (B, r, n)
        A batch of ``B`` reduced-coordinate tensors, all with the same
        ``(r, n)`` shape.
    multi_indices : (M, r)  int
        Shared grlex multi-indices.
    n_reorth : int

    Returns
    -------
    Psi_batch : (B, M, n)
    H_batch   : (B, M, M)
    ell_arr   : (M,)  int32  (shared)
    parent_arr: (M,)  int32  (shared)
    """
    multi_indices = np.asarray(multi_indices, dtype=np.int32)
    if multi_indices.ndim != 2 or int(multi_indices[0].sum()) != 0:
        raise ValueError(
            "multi_indices must be 2D with the zero multi-index first."
        )

    ell_np, parent_np = compute_parent_info(multi_indices)
    ell_arr = jnp.asarray(ell_np)
    parent_arr = jnp.asarray(parent_np)

    fn = partial(
        _arnoldi_build_jit,
        ell_arr=ell_arr,
        parent_arr=parent_arr,
        n_reorth=int(n_reorth),
    )
    Psi_batch, H_batch = jax.vmap(fn)(S_hat_batch)
    return Psi_batch, H_batch, ell_arr, parent_arr


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def arnoldi_orthonormality_error(Psi: jnp.ndarray) -> float:
    """``||(1/n) Psi Psi^T - I||_F`` for a built Psi."""
    M, n = Psi.shape
    G = (1.0 / n) * (Psi @ Psi.T)
    return float(jnp.linalg.norm(G - jnp.eye(M)))