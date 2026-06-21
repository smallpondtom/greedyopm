"""Quadratic manifold with leading-r POD modes + group-sparse column selection.

This is the Geelen-Wright-Willcox quadratic-manifold baseline
(github.com/geelenr/quad_manifold): the basis is the *leading* r POD modes
(no greedy mode selection), and the quadratic lifting matrix is found by the
group-sparse regularized least-squares problem

    Vbar = argmin 1/2 || Vbar P - E ||_F^2 + gamma_sp * sum_k || Vbar[:,k] ||_2,

where P = quad-features(Shat) and E = (I - V V^T) S_tilde is the POD residual.
The group penalty drives whole columns of Vbar (i.e. whole quadratic features)
to zero, selecting a subset of features.  We solve it with FISTA (accelerated
proximal gradient, column-wise group soft-thresholding), then *refit* the
selected columns by Tikhonov-regularized least squares to remove shrinkage bias
(the same select-then-refit policy used by the greedy-OPM lifting).

The quadratic feature map is imported from ``qm.quadmani`` so the features match
Greedy-QM exactly.
"""

from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from qm.quadmani import quadratic_mapping


def _fista_group_lasso(G, EPt, lam, L, n_iter):
    """min_{Vbar} 1/2 tr(Vbar G Vbar^T) - tr(Vbar EPt^T) + lam sum_k||Vbar[:,k]||.

    G = P P^T (q x q), EPt = E P^T (d x q).  Returns Vbar (d x q).
    """
    d = EPt.shape[0]
    q = G.shape[0]
    Vbar = jnp.zeros((d, q))
    Y = Vbar
    t = 1.0
    thr = lam / L

    def prox(Z):
        col = jnp.linalg.norm(Z, axis=0)                  # (q,)
        scale = jnp.maximum(0.0, 1.0 - thr / jnp.maximum(col, 1e-300))
        return Z * scale[None, :]

    def body(carry, _):
        Vbar, Y, t = carry
        grad = Y @ G - EPt                                # (d,q)
        Vbar_new = prox(Y - grad / L)
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        Y_new = Vbar_new + ((t - 1.0) / t_new) * (Vbar_new - Vbar)
        return (Vbar_new, Y_new, t_new), None

    (Vbar, _, _), _ = jax.lax.scan(body, (Vbar, Y, t), None, length=n_iter)
    return Vbar


class GeelenQM:
    """Leading-r POD QM with group-sparse quadratic feature selection."""

    def __init__(self, r, *, gamma_sp_rel=0.1, gamma=1e-8, n_iter=400,
                 active_tol=1e-8, center=True):
        self.r = r
        self.gamma_sp_rel = gamma_sp_rel    # threshold relative to max column corr.
        self.gamma = gamma                  # Tikhonov for the refit
        self.n_iter = n_iter
        self.active_tol = active_tol
        self.center = center
        self._fitted = False

    def fit(self, S):
        S = jnp.asarray(S)
        d, n = S.shape
        self.s_ref_ = jnp.mean(S, axis=1, keepdims=True) if self.center \
            else jnp.zeros((d, 1))
        St = S - self.s_ref_

        U, _, _ = jnp.linalg.svd(St, full_matrices=False)
        self.V_ = U[:, : self.r]                          # leading-r POD
        Shat = self.V_.T @ St                             # (r,n)
        E = St - self.V_ @ Shat                           # (d,n) residual

        P = quadratic_mapping(Shat)                       # (q,n)
        self.q_ = int(P.shape[0])
        G = P @ P.T                                       # (q,q)
        EPt = E @ P.T                                     # (d,q)

        L = float(jnp.linalg.eigvalsh(G)[-1]) + 1e-30     # Lipschitz constant
        col_corr = jnp.linalg.norm(EPt, axis=0)           # (q,)
        lam = float(self.gamma_sp_rel * jnp.max(col_corr))

        Vbar = _fista_group_lasso(G, EPt, lam, L, self.n_iter)
        col_norm = jnp.linalg.norm(Vbar, axis=0)
        active = jnp.where(col_norm > self.active_tol * jnp.max(col_norm))[0]
        if active.size == 0:                              # keep the strongest
            active = jnp.array([int(jnp.argmax(col_corr))])
        self.active_ = np.asarray(active)

        # Refit the selected columns (debiased, Tikhonov).
        P_a = P[active]                                   # (|A|,n)
        Gram = P_a @ P_a.T + self.gamma * jnp.eye(active.size)
        self.Vbar_ = (E @ P_a.T) @ jnp.linalg.inv(Gram)   # (d,|A|)
        self._fitted = True
        return self

    def encode(self, S):
        return self.V_.T @ (jnp.asarray(S) - self.s_ref_)

    def decode(self, Shat):
        Shat = jnp.asarray(Shat)
        P = quadratic_mapping(Shat)[self.active_]
        return self.s_ref_ + self.V_ @ Shat + self.Vbar_ @ P

    def reconstruct(self, S):
        return self.decode(self.encode(S))

    def reconstruction_error(self, S, *, relative=True):
        S = jnp.asarray(S)
        approx = self.reconstruct(S)
        err = float(jnp.linalg.norm(S - approx, "fro"))
        if relative:
            return err / float(jnp.linalg.norm(S - self.s_ref_, "fro"))
        return err

    @property
    def n_active(self):
        return int(self.active_.size)
