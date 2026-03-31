from __future__ import annotations 

import functools
import jax
import jax.numpy as jnp
from jax import lax

@functools.partial(jax.jit, static_argnames=("max_deg",))
def _chebyshev_1d(xi: jnp.ndarray, max_deg: int) -> jnp.ndarray:
    """Normalised Chebyshev (1st kind) polynomials T_0, ..., T_{max_deg}.

    Normalisation: int T_i T_j w(x) dx = delta_{ij}, w = 1/(pi sqrt(1-x^2)).
    """
    xi = jnp.atleast_1d(xi)
    vals = jnp.zeros((max_deg + 1,) + xi.shape, dtype=xi.dtype)
    vals = vals.at[0].set(jnp.ones_like(xi))

    if max_deg == 0:
        return vals

    vals = vals.at[1].set(xi)

    def body(n, vals):
        pn = 2.0 * xi * vals[n] - vals[n - 1]
        vals = vals.at[n + 1].set(pn)
        return vals

    vals = lax.fori_loop(1, max_deg, body, vals)

    # Normalise: T_0 -> 1, T_n -> sqrt(2) T_n  for n >= 1
    norms = jnp.where(jnp.arange(max_deg + 1) >= 1, jnp.sqrt(2.0), 1.0)
    return vals * norms[:, None]
