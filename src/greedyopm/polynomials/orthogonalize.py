from __future__ import annotations

import jax 
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Literal, Dict

from greedyopm.polynomials.multindex import (
    generate_multi_indices,
    graded_lex_multi_indices,
    degree_slices
)
from greedyopm.polynomials.monomial import evaluate_monomials
from greedyopm.polynomials.classical_tensor import evaluate_classical_tensor

@jax.jit
def orthogonalize_qr(Phi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """QR-based orthogonalization (Vandermonde with Arnoldi).

    Given Phi in R^{m x k}, compute QR of (1/sqrt(k)) Phi^T,
    then Psi = sqrt(k) Q^T so that (1/k) Psi Psi^T = I.

    Returns (Psi, R).
    """
    m, k = Phi.shape
    F_scaled = Phi.T / jnp.sqrt(k)           # (k, m)
    Q, R = jnp.linalg.qr(F_scaled, mode="reduced")  # Q: (k, m), R: (m, m)

    # Fix sign convention: ensure diagonal of R is positive
    signs = jnp.sign(jnp.diag(R))
    signs = jnp.where(signs == 0, 1.0, signs)
    Q = Q * signs[None, :]
    R = signs[:, None] * R

    Psi = jnp.sqrt(k) * Q.T                 # (m, k)
    return Psi, R


@jax.jit
def orthogonalize_cholesky(Phi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Cholesky-based orthogonalization.

    Given Phi in R^{m x k}, compute Psi = L^{-1} Phi such that (1/k) Psi Psi^T = I.

    Returns (Psi, L).
    """
    m, k = Phi.shape
    G = (1.0 / k) * (Phi @ Phi.T)           # (m, m) Gram matrix
    G = G + 1e-14 * jnp.eye(m)               # regularisation
    L = jnp.linalg.cholesky(G)               # (m, m) lower triangular
    Psi = jax.scipy.linalg.solve_triangular(L, Phi, lower=True)
    return Psi, L


def orthogonalize_hierarchical(
    S_hat: jnp.ndarray,
    p: int,
    *,
    method: Literal["qr", "cholesky"] = "qr",
) -> Tuple[jnp.ndarray, Dict[int, jnp.ndarray], Dict[int, np.ndarray], jnp.ndarray]:
    """Build orthonormal polynomial features from monomials, degree by degree.

    Parameters
    ----------
    S_hat : (r, k)
    p : maximum polynomial degree
    method : "qr" or "cholesky"

    Returns
    -------
    Psi : (m_total, k)  — full orthonormal evaluation matrix
    P_by_degree : {q: Psi_q} with Psi_q of shape (m_q, k)
    indices_by_degree : {q: multi_indices_q}
    RL_inv_T : (m_total, m_total) — either inv(R^T) or inv(L)
    """
    r = S_hat.shape[0]
    all_indices_np = graded_lex_multi_indices(r, p, min_degree=0)
    all_indices = jnp.array(all_indices_np)

    Phi = evaluate_monomials(S_hat, all_indices)

    if method == "qr":
        Psi, R = orthogonalize_qr(Phi)
        RL_inv_T = jnp.linalg.inv(R.T)
    else:
        Psi, L = orthogonalize_cholesky(Phi)
        RL_inv_T = jnp.linalg.inv(L)

    # Split by degree
    slices = degree_slices(r, p)
    P_by_degree = {q: Psi[sl] for q, sl in slices.items()}
    indices_by_degree: Dict[int, np.ndarray] = {}
    for q in range(p + 1):
        indices_by_degree[q] = generate_multi_indices(r, q)

    return Psi, P_by_degree, indices_by_degree, RL_inv_T


def orthogonalize_hybrid(
    S_hat: jnp.ndarray,
    p: int,
    family: str,
    *,
    method: Literal["qr", "cholesky"] = "qr",
) -> Tuple[jnp.ndarray, Dict[int, jnp.ndarray], Dict[int, np.ndarray], jnp.ndarray]:
    """Hybrid classical-empirical orthogonalization.

    1. Evaluate classical tensor-product polynomials (approximately orthogonal).
    2. Apply QR/Cholesky to enforce exact empirical orthonormality.

    Parameters
    ----------
    S_hat : (r, k)
    p : maximum polynomial degree
    family : {"legendre", "chebyshev", "hermite"}
    method : "qr" or "cholesky"

    Returns
    -------
    Psi, P_by_degree, indices_by_degree, RL_inv_T  (same layout as above)
    """
    r = S_hat.shape[0]
    all_indices_np = graded_lex_multi_indices(r, p, min_degree=0)
    all_indices = jnp.array(all_indices_np)
    max_deg = int(np.max(all_indices_np))

    Phi_tilde = evaluate_classical_tensor(S_hat, all_indices, family, max_deg)

    if method == "qr":
        Psi, R = orthogonalize_qr(Phi_tilde)
        RL_inv_T = jnp.linalg.inv(R.T)
    else:
        Psi, L = orthogonalize_cholesky(Phi_tilde)
        RL_inv_T = jnp.linalg.inv(L)

    slices = degree_slices(r, p)
    P_by_degree = {q: Psi[sl] for q, sl in slices.items()}
    indices_by_degree: Dict[int, np.ndarray] = {}
    for q in range(p + 1):
        indices_by_degree[q] = generate_multi_indices(r, q)

    return Psi, P_by_degree, indices_by_degree, RL_inv_T
