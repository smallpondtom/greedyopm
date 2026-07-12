# Greedy Orthogonal Polynomial Manifold

## Stability regularization (Jacobian / Lipschitz control)

`GreedyOPM` supports out-of-distribution stabilization in the style of
Bensalah, Nouy & Soffo (arXiv:2502.05088), adapted to the OPM structure; see
`paper/.../stability_widths_note.tex` for the theory.

```python
model = GreedyOPM(
    r=4, s=16, p=3,
    mu=1e-2,            # Jacobian penalty weight (0 = original algorithm)
    grad_dilation=0.25, # probe gradients at (1 + delta) * s_hat (OOD collar)
    n_grad_samples=64,  # subsample probe points in the greedy loop
    stability_aware_activation=False,  # True: weighted threshold
                        # ||r_k|| > (gamma_sp/n) sqrt(c_k/n); hard-removes
                        # features with huge Dirichlet energy
).fit(S)

model.lipschitz_certificate(dilation=0.25)  # a posteriori bound L_hat
model.decoder_jacobian_norms(S_hat_pts)     # pointwise ||J_Gamma||_2
```

The per-feature gain becomes `(n||r_k|| - gamma_sp)^2 / (n + gamma + mu E_k)`
where `E_k` is the empirical Dirichlet energy of feature `k`, computed by the
differentiated Arnoldi recurrence (`arnoldi_eval_grad`); the lifting columns
are shrunk by `n / (n + gamma + mu E_k)`. A useful scale for `mu` is
`n / median_k(E_k)`. See `experiments/stability_width_study.py` for the OOD
sweep and the Kolmogorov-width matching study.
