from __future__ import annotations

import functools 
import jax 
import jax.numpy as jnp

from greedyopm.polynomials.legendre import _legendre_1d
from greedyopm.polynomials.chebyshev import _chebyshev_1d
from greedyopm.polynomials.hermite import _hermite_1d

_FAMILY_DISPATCH = {
    "legendre": _legendre_1d,
    "chebyshev": _chebyshev_1d,
    "hermite": _hermite_1d,
}

@functools.partial(jax.jit, static_argnames=("family", "max_deg"))
def evaluate_classical_tensor(
    S_hat: jnp.ndarray,
    multi_indices: jnp.ndarray,
    family: str,
    max_deg: int,
) -> jnp.ndarray:
    """Evaluate tensor-product classical polynomials at each snapshot.

    Parameters
    ----------
    S_hat : (r, n)
    multi_indices : (m, r)  — JAX int array
    family : {"legendre", "chebyshev", "hermite"}
    max_deg : maximum polynomial degree (static for JIT)

    Returns
    -------
    Phi_tilde : (m, n)
    """
    r, k = S_hat.shape

    # --- Step 1: rescale coordinates to classical domain ---
    if family == "hermite":
        mu = jnp.mean(S_hat, axis=1, keepdims=True)       # (r, 1)
        sigma = jnp.std(S_hat, axis=1, keepdims=True)     # (r, 1)
        sigma = jnp.maximum(sigma, 1e-15)
        xi = (S_hat - mu) / sigma                         # (r, n)
    else:
        a = jnp.min(S_hat, axis=1, keepdims=True)         # (r, 1)
        b = jnp.max(S_hat, axis=1, keepdims=True)         # (r, 1)
        rng = jnp.maximum(b - a, 1e-15)
        xi = 2.0 * (S_hat - a) / rng - 1.0                # (r, n)
        if family == "chebyshev":
            xi = xi * 0.999  # contract away from boundary

    # --- Step 2: precompute univariate polynomial values ---
    # Evaluate all univariate polynomials for each dimension.
    # For each dimension d, evaluate the chosen family at xi[d, :].
    # Result: uni_vals[d, n, j] = phi_n(xi_{d,j})
    eval_fn = functools.partial(_FAMILY_DISPATCH[family], max_deg=max_deg)

    # Vectorise over dimensions using vmap
    # eval_fn expects (N_points,) and returns (max_deg+1, N_points)
    uni_vals = jax.vmap(eval_fn, in_axes=(0,), out_axes=0)(xi)
    # uni_vals: (r, max_deg+1, k)

    # --- Step 3: tensor products via advanced indexing ---
    # For multi-index alpha = (a1, ..., ar), the tensor product is
    #   prod_{d=1}^{r} uni_vals[d, alpha_d, :]
    # multi_indices: (m, r)
    # We gather uni_vals[d, multi_indices[i, d], j] for all (i, d, j)
    # then take product over d.

    dims = jnp.arange(r)  # (r,)
    # Gather: for each (i, d), get uni_vals[d, multi_indices[i, d], :]
    # → shape (m, r, k), then prod over axis=1
    gathered = uni_vals[
        dims[None, :],                   # (1, r) → broadcast to (m, r)
        multi_indices,                   # (m, r)
    ]  # → (m, r, n)

    return jnp.prod(gathered, axis=1)  # (m, n)
