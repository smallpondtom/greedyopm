from __future__ import annotations

import jax
import jax.numpy as jnp

@jax.jit
def evaluate_monomials(S_hat: jnp.ndarray,
                       multi_indices: jnp.ndarray) -> jnp.ndarray:
    """Evaluate monomials  s_hat^alpha  at each snapshot.

    Parameters
    ----------
    S_hat : (r, n)
    multi_indices : (m, r) — JAX integer array

    Returns
    -------
    Phi : (m, n)
    """
    # S_hat[d, j]^alpha[i, d]  →  prod over d
    # Use log-space for numerical stability with high degrees
    # S_hat: (r, n), multi_indices: (m, r)
    # Broadcast:  S_hat[None, :, :] ** multi_indices[:, :, None]  → (m, r, n)
    # Then prod over axis 1
    return jnp.prod(
        S_hat[None, :, :] ** multi_indices[:, :, None],
        axis=1,
    )  # (m, n)
