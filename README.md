# Greedy Orthogonal Polynomial Manifold

## Stability regularization (Jacobian / Lipschitz control)

`GreedyOPM` supports out-of-distribution stabilization in the style of
Bensalah, Nouy & Soffo (arXiv:2502.05088), adapted to the OPM structure; see
the paper (`paper/.../sections/method.tex`) for the theory.

```python
model = GreedyOPM(
    r=4, s=16, p=3,
    mu=1e-2,            # Jacobian penalty weight (0 = original algorithm)
    grad_dilation=0.25, # probe gradients at (1 + delta) * s_hat (OOD collar)
    n_grad_samples=64,  # subsample probe points in the greedy loop
    budget=5.0,         # optional: fit() enforces L_hat <= budget (mu-loop)
).fit(S)

model.certificate_                          # a posteriori bound, set by fit()
model.lipschitz_certificate(dilation=0.25)  # re-evaluate at another dilation
model.decoder_jacobian_norms(S_hat_pts)     # pointwise ||J_Gamma||_2
model.enforce_budget(L=5.0, dilation=0.25)  # outer mu-loop, callable post hoc
model.polish_exact(mu=1e2)                  # OPTIONAL exact tr(V K V^T) solve
```

The per-feature gain is `(n||r_k|| - gamma_sp)_+^2 / c_k` with
`c_k = n + gamma + mu E_k`, where `E_k` is the empirical Dirichlet energy of
feature `k`, computed by the differentiated Arnoldi recurrence
(`arnoldi_eval_grad`); the lifting columns are the group soft-thresholded
`(1 - gamma_sp/(n||r_k||))_+ (n/c_k) r_k` of the one-step theorem. A useful
scale for `mu` is `n / median_k(E_k)`, and `enforce_budget` finds it
automatically for a prescribed Lipschitz budget `L`. See
`experiments/stability_width_study.py` for the OOD sweep.
