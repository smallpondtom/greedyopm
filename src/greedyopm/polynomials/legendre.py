"""Empirically normalised univariate Legendre polynomials on [-1, 1]."""

from __future__ import annotations 

import functools
import jax
import jax.numpy as jnp
from jax import lax

@functools.partial(jax.jit, static_argnames=("max_deg",))
def _legendre_1d(xi: jnp.ndarray, max_deg: int) -> jnp.ndarray:
    """Normalised Legendre polynomials P_0, ..., P_{max_deg} evaluated at xi.

    Normalisation: int_{-1}^{1} P_i P_j dx/2 = delta_{ij}.
    xi : (N,) or scalar broadcast
    Returns : (max_deg+1, *xi.shape)
    """
    xi = jnp.atleast_1d(xi)
    vals = jnp.zeros((max_deg + 1,) + xi.shape, dtype=xi.dtype)
    vals = vals.at[0].set(jnp.ones_like(xi))

    if max_deg == 0:
        # Normalise degree 0
        return vals * jnp.sqrt(jnp.arange(max_deg + 1) * 2 + 1)[:, None]

    vals = vals.at[1].set(xi)

    def body(n, vals):
        pn = ((2 * n + 1) * xi * vals[n] - n * vals[n - 1]) / (n + 1)
        vals = vals.at[n + 1].set(pn)
        return vals

    vals = lax.fori_loop(1, max_deg, body, vals)

    # Normalise: P_n^tilde = sqrt(2n+1) P_n
    norms = jnp.sqrt(jnp.arange(max_deg + 1) * 2.0 + 1.0)
    return vals * norms[:, None]

