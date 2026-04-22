"""QR/Cholesky-based orthogonalisation of polynomial feature matrices.

These routines are provided as fallback / comparison paths.  For the greedy
algorithm, the Arnoldi-based construction in :mod:`arnoldi` and
:mod:`incremental_arnoldi` is recommended (see the corresponding sections of
the paper).

Given  Phi in R^{M x n},  we want  Psi = factor^{-} Phi  with
``(1/n) Psi Psi^T = I_M`` and ``factor`` lower- or upper-triangular.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Literal, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .classical_tensor import evaluate_classical_tensor
from .monomial import evaluate_monomials
from .multindex import (
    generate_multi_indices,
    graded_lex_multi_indices,
)


# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------


def _fix_sign_convention(
    Q: jnp.ndarray, R: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Ensure positive diagonal of R (and adjust Q accordingly)."""
    signs = jnp.sign(jnp.diag(R))
    signs = jnp.where(signs == 0, 1.0, signs)
    return Q * signs[None, :], signs[:, None] * R


# ---------------------------------------------------------------------------
# Base QR / Cholesky (no refinement)
# ---------------------------------------------------------------------------


@jax.jit
def orthogonalize_qr(Phi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Thin QR of (1/sqrt(n)) Phi^T.  Returns ``(Psi, R)`` with Psi = sqrt(n) Q^T."""
    M, n = Phi.shape
    Q, R = jnp.linalg.qr(Phi.T / jnp.sqrt(n), mode="reduced")
    Q, R = _fix_sign_convention(Q, R)
    return jnp.sqrt(n) * Q.T, R


@jax.jit
def orthogonalize_cholesky(
    Phi: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Cholesky factor of the empirical Gram matrix.  Returns ``(Psi, L)``."""
    M, n = Phi.shape
    G = (1.0 / n) * (Phi @ Phi.T) + 1e-14 * jnp.eye(M)
    L = jnp.linalg.cholesky(G)
    Psi = jax.scipy.linalg.solve_triangular(L, Phi, lower=True)
    return Psi, L


# ---------------------------------------------------------------------------
# Iterative refinement ("twice is enough")
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(1,))
def orthogonalize_qr_refined(
    Phi: jnp.ndarray, n_reorth: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """QR with ``n_reorth`` extra refinement passes on the Q factor."""
    M, n = Phi.shape
    Q, R = jnp.linalg.qr(Phi.T / jnp.sqrt(n), mode="reduced")
    Q, R = _fix_sign_convention(Q, R)
    R_accum = R

    if n_reorth > 0:
        def step(carry, _):
            Q_in, R_acc = carry
            Q_new, R_new = jnp.linalg.qr(Q_in, mode="reduced")
            Q_new, R_new = _fix_sign_convention(Q_new, R_new)
            return (Q_new, R_new @ R_acc), None
        (Q, R_accum), _ = jax.lax.scan(
            step, (Q, R_accum), None, length=n_reorth,
        )

    return jnp.sqrt(n) * Q.T, R_accum


@partial(jax.jit, static_argnums=(1,))
def orthogonalize_cholesky_refined(
    Phi: jnp.ndarray, n_reorth: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Cholesky with ``n_reorth`` extra refinement passes."""
    M, n = Phi.shape
    G = (1.0 / n) * (Phi @ Phi.T) + 1e-14 * jnp.eye(M)
    L = jnp.linalg.cholesky(G)
    Psi = jax.scipy.linalg.solve_triangular(L, Phi, lower=True)
    L_accum = L

    if n_reorth > 0:
        def step(carry, _):
            Psi_in, L_acc = carry
            G_new = (1.0 / n) * (Psi_in @ Psi_in.T) + 1e-14 * jnp.eye(M)
            L_new = jnp.linalg.cholesky(G_new)
            Psi_new = jax.scipy.linalg.solve_triangular(L_new, Psi_in, lower=True)
            return (Psi_new, L_new @ L_acc), None
        (Psi, L_accum), _ = jax.lax.scan(
            step, (Psi, L_accum), None, length=n_reorth,
        )

    return Psi, L_accum


# ---------------------------------------------------------------------------
# Safe / rank-aware inverses
# ---------------------------------------------------------------------------


def _safe_inverse_factor_qr(
    R: jnp.ndarray, rank_tol: float = 1e-12,
) -> Tuple[jnp.ndarray, int]:
    """Rank-truncated R^{-T}, padded with zeros to ``(M, M)``."""
    diag_R = jnp.abs(jnp.diag(R))
    threshold = rank_tol * jnp.max(diag_R)
    rank = int(jnp.sum(diag_R > threshold))

    M = R.shape[0]
    R_trunc = R[:rank, :rank]
    R_inv_T_trunc = jnp.linalg.inv(R_trunc.T)

    if rank < M:
        R_inv_T = jnp.zeros((M, M))
        R_inv_T = R_inv_T.at[:rank, :rank].set(R_inv_T_trunc)
    else:
        R_inv_T = R_inv_T_trunc
    return R_inv_T, rank


def _safe_inverse_factor_cholesky(
    L: jnp.ndarray, rank_tol: float = 1e-12,
) -> Tuple[jnp.ndarray, int]:
    """Rank-truncated L^{-1}, padded with zeros to ``(M, M)``."""
    diag_L = jnp.abs(jnp.diag(L))
    threshold = rank_tol * jnp.max(diag_L)
    rank = int(jnp.sum(diag_L > threshold))

    M = L.shape[0]
    L_trunc = L[:rank, :rank]
    L_inv_trunc = jnp.linalg.inv(L_trunc)

    if rank < M:
        L_inv = jnp.zeros((M, M))
        L_inv = L_inv.at[:rank, :rank].set(L_inv_trunc)
    else:
        L_inv = L_inv_trunc
    return L_inv, rank


# ---------------------------------------------------------------------------
# High-level wrappers (monomial / classical starting basis).  Useful for the
# GreedyFixedOPM path that doesn't interleave with the greedy loop.
# ---------------------------------------------------------------------------


def orthogonalize_hierarchical(
    S_hat: jnp.ndarray,
    p: int,
    *,
    method: Literal["qr", "cholesky"] = "qr",
    n_reorth: int = 0,
    rank_tol: float = 0.0,
) -> Tuple[jnp.ndarray, Dict[int, jnp.ndarray], Dict[int, np.ndarray], jnp.ndarray]:
    """Orthonormalise monomial features up to degree ``p``."""
    r = S_hat.shape[0]
    all_indices_np = graded_lex_multi_indices(r, p, min_degree=0)
    Phi = evaluate_monomials(S_hat, jnp.array(all_indices_np))
    return _finalise(Phi, r, p, method, n_reorth, rank_tol,
                     S_hat.shape[1])


def orthogonalize_hybrid(
    S_hat: jnp.ndarray,
    p: int,
    family: str,
    *,
    method: Literal["qr", "cholesky"] = "qr",
    n_reorth: int = 0,
    rank_tol: float = 0.0,
) -> Tuple[jnp.ndarray, Dict[int, jnp.ndarray], Dict[int, np.ndarray], jnp.ndarray]:
    """Orthonormalise tensor-product classical polynomials up to degree ``p``."""
    r = S_hat.shape[0]
    all_indices_np = graded_lex_multi_indices(r, p, min_degree=0)
    max_deg = int(np.max(all_indices_np))
    Phi = evaluate_classical_tensor(
        S_hat, jnp.array(all_indices_np), family, max_deg,
    )
    return _finalise(Phi, r, p, method, n_reorth, rank_tol,
                     S_hat.shape[1])


def _finalise(
    Phi: jnp.ndarray,
    r: int,
    p: int,
    method: str,
    n_reorth: int,
    rank_tol: float,
    n: int,
):
    """Shared post-processing for the two public entry points above."""
    from .multindex import degree_slices_for_degrees

    if method == "qr":
        if n_reorth > 0:
            Psi, R = orthogonalize_qr_refined(Phi, n_reorth)
        else:
            Psi, R = orthogonalize_qr(Phi)
        if rank_tol > 0.0:
            RL_inv_T, rank = _safe_inverse_factor_qr(R, rank_tol)
            Psi = Psi[:rank]
        else:
            RL_inv_T = jnp.linalg.inv(R.T)
    else:
        if n_reorth > 0:
            Psi, L = orthogonalize_cholesky_refined(Phi, n_reorth)
        else:
            Psi, L = orthogonalize_cholesky(Phi)
        if rank_tol > 0.0:
            RL_inv_T, rank = _safe_inverse_factor_cholesky(L, rank_tol)
            Psi = Psi[:rank]
        else:
            RL_inv_T = jnp.linalg.inv(L)

    # Organise by degree
    degrees_nl = list(range(2, p + 1))
    slices = degree_slices_for_degrees(r, degrees_nl, include_linear=True)
    m_eff = Psi.shape[0]
    P_by_degree: Dict[int, jnp.ndarray] = {}
    for q, sl in slices.items():
        start = min(sl.start, m_eff)
        stop = min(sl.stop, m_eff)
        P_by_degree[q] = Psi[start:stop] if start < stop else jnp.zeros((0, n))

    indices_by_degree: Dict[int, np.ndarray] = {
        q: generate_multi_indices(r, q) for q in range(p + 1)
    }
    return Psi, P_by_degree, indices_by_degree, RL_inv_T


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def check_orthonormality(Psi: jnp.ndarray, label: str = "") -> float:
    """``||(1/n) Psi Psi^T - I||_F`` as a print-friendly diagnostic."""
    M, n = Psi.shape
    err = float(jnp.linalg.norm((1.0 / n) * (Psi @ Psi.T) - jnp.eye(M)))
    tag = f" [{label}]" if label else ""
    print(f"  Orthonormality error{tag} = {err:.3e}")
    return err