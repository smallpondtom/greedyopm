"""Empirically normalised univariate probabilist's Hermite polynomials."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax


@functools.partial(jax.jit, static_argnames=("max_deg",))
def _hermite_1d(xi: jnp.ndarray, max_deg: int) -> jnp.ndarray:
    """Orthonormal probabilist's Hermite polynomials  He_0, ..., He_{max_deg}.

    Normalisation:
        int He_i He_j exp(-x^2 / 2) / sqrt(2 pi) dx = delta_{ij}.
    """
    xi = jnp.atleast_1d(xi)
    vals = jnp.zeros((max_deg + 1,) + xi.shape, dtype=xi.dtype)
    vals = vals.at[0].set(jnp.ones_like(xi))

    if max_deg == 0:
        return vals

    vals = vals.at[1].set(xi)

    def body(n, vals):
        pn = xi * vals[n] - n * vals[n - 1]
        vals = vals.at[n + 1].set(pn)
        return vals

    vals = lax.fori_loop(1, max_deg, body, vals)

    # He_n -> He_n / sqrt(n!)
    log_factorials = jnp.cumsum(
        jnp.log(jnp.maximum(jnp.arange(max_deg + 1, dtype=xi.dtype), 1.0))
    )
    log_factorials = log_factorials.at[0].set(0.0)
    norms = jnp.exp(-0.5 * log_factorials)
    return vals * norms[:, None]