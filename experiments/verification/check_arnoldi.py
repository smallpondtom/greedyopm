#%%
import numpy as np, jax.numpy as jnp
from greedyopm.polynomials.multindex import graded_lex_multi_indices
from greedyopm.polynomials.arnoldi import (
    arnoldi_build, arnoldi_eval, arnoldi_orthonormality_error,
)

#%%
rng = np.random.default_rng(0)
r, n, p = 6, 500, 2
S_hat = jnp.asarray(rng.standard_normal((r, n)))
mi = graded_lex_multi_indices(r, p, min_degree=0)

#%%
Psi, basis = arnoldi_build(S_hat, mi, n_reorth=1)
print("ortho err:", arnoldi_orthonormality_error(Psi))           # ~1e-14

#%%
# Out-of-sample consistency: evaluating at the training points must
# reproduce Psi to machine precision.
Psi_check = arnoldi_eval(S_hat, basis)
print("eval err :", float(jnp.linalg.norm(Psi - Psi_check)))     # ~1e-12
# %%
