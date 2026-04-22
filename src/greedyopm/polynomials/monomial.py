"""Monomial evaluation:  s_hat^alpha  for all alpha in a multi-index array."""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def evaluate_monomials(
    S_hat: jnp.ndarray,
    multi_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Monomial features at each snapshot.

    Parameters
    ----------
    S_hat : (r, n)
    multi_indices : (m, r)  int

    Returns
    -------
    Phi : (m, n) where  Phi[i, j] = prod_d S_hat[d, j] ** multi_indices[i, d].
    """
    # Broadcast to (m, r, n) and take product along the r-axis.
    return jnp.prod(
        S_hat[None, :, :] ** multi_indices[:, :, None], axis=1,
    )