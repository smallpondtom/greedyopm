import jax
import jax.numpy as jnp
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=32)
def _quad_indices(n: int):
    """Precompute lower-triangular index pairs (cached, concrete arrays)."""
    i, j = np.tril_indices(n)
    return jnp.asarray(i, dtype=jnp.int32), jnp.asarray(j, dtype=jnp.int32)


@lru_cache(maxsize=32)
def _cubic_indices(n: int):
    """Precompute sorted triple indices i <= j <= k (cached, concrete arrays)."""
    num = n * (n + 1) * (n + 2) // 6
    i_arr = np.empty(num, dtype=np.int32)
    j_arr = np.empty(num, dtype=np.int32)
    k_arr = np.empty(num, dtype=np.int32)
    c = 0
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                i_arr[c], j_arr[c], k_arr[c] = i, j, k
                c += 1
    return jnp.asarray(i_arr), jnp.asarray(j_arr), jnp.asarray(k_arr)

@jax.jit
def quadratic_mapping(x: jnp.ndarray) -> jnp.ndarray:
    """Unique quadratic products x_i * x_j for i >= j.

    Args:
        x: shape (n,) or (batch, n).

    Returns:
        shape (n*(n+1)//2,) or (batch, n*(n+1)//2).
    """
    n = x.shape[-1]
    ii, jj = _quad_indices(n)
    if x.ndim == 1:
        return x[ii] * x[jj]
    return x[:, ii] * x[:, jj]

@jax.jit
def cubic_mapping(x: jnp.ndarray) -> jnp.ndarray:
    """Unique cubic products x_i * x_j * x_k for i <= j <= k.

    Args:
        x: shape (n,) or (batch, n).

    Returns:
        shape (n*(n+1)*(n+2)//6,) or (batch, n*(n+1)*(n+2)//6).
    """
    n = x.shape[-1]
    ii, jj, kk = _cubic_indices(n)
    if x.ndim == 1:
        return x[ii] * x[jj] * x[kk]
    return x[:, ii] * x[:, jj] * x[:, kk]