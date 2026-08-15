"""Baseline manifolds for the advection-diffusion comparison.

Self-contained JAX implementations of the three baselines:

* :class:`POD`      -- leading-r linear projection (the linear reference).
* :class:`GreedyQM` -- greedy quadratic manifold of Schwerdtner &
  Peherstorfer (2024): greedy mode selection from an s-pool scored by the
  ridge-regularized quadratic-monomial fit of the residual, with the full
  quadratic shell active.
* :class:`GeelenQM` -- quadratic manifold of Geelen, Wright & Willcox
  (2023): leading-r POD modes, group-sparse selection of quadratic-monomial
  columns by FISTA, then a debiased Tikhonov refit of the active columns.

All models expose the same ``encode`` / ``decode`` / ``reconstruct`` /
``reconstruction_error`` API as the OPM models.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Quadratic monomial features (shared by both QM baselines)
# ---------------------------------------------------------------------------


@jax.jit
def quadratic_features(S_hat: jnp.ndarray) -> jnp.ndarray:
    """Distinct quadratic monomials s_i s_j (i <= j) of each snapshot.

    Arguments
    ---------
    S_hat : (r, n)

    Returns
    -------
    P : (r (r+1) / 2, n)
    """
    r = S_hat.shape[0]
    return jnp.concatenate(
        [S_hat[i] * S_hat[: i + 1] for i in range(r)], axis=0,
    )


@jax.jit
def _ridge_fit_error(P: jnp.ndarray, B: jnp.ndarray, reg: float) -> jnp.ndarray:
    """Residual norm of the ridge fit  min_X ||X P - B||_F  (SVD filter).

    Matches the Tikhonov filter ``sigma / (sigma^2 + reg^2)`` of the
    reference Greedy-QM implementation.
    """
    U, sv, Vt = jnp.linalg.svd(P, full_matrices=False)
    filt = sv / (sv ** 2 + reg ** 2)
    X = ((B @ Vt.T) * filt[None, :]) @ U.T
    return jnp.linalg.norm(B - X @ P, "fro")


# ---------------------------------------------------------------------------
# POD
# ---------------------------------------------------------------------------


class POD:
    """Leading-r POD projection (linear reference).

    Arguments
    ---------
    r : int
        Reduced dimension.
    center : bool
        Whether to subtract the snapshot mean before the SVD.
    """

    def __init__(self, r: int, *, center: bool = True):
        self.r = int(r)
        self.center = bool(center)

    def fit(self, S: jnp.ndarray) -> "POD":
        """Fit to snapshot data ``S`` of shape ``(d, n)``."""
        S = jnp.asarray(S)
        d, _ = S.shape
        self.s_ref_ = (
            jnp.mean(S, axis=1, keepdims=True)
            if self.center else jnp.zeros((d, 1), dtype=S.dtype)
        )
        U, _, _ = jnp.linalg.svd(S - self.s_ref_, full_matrices=False)
        self.U_r_ = U[:, : self.r]
        return self

    def encode(self, S: jnp.ndarray) -> jnp.ndarray:
        """Linear encoding ``s_hat = V^T (s - s_ref)``."""
        return self.U_r_.T @ (jnp.asarray(S) - self.s_ref_)

    def decode(self, S_hat: jnp.ndarray) -> jnp.ndarray:
        """Linear decoding ``s_ref + V s_hat``."""
        return self.s_ref_ + self.U_r_ @ jnp.asarray(S_hat)

    def reconstruct(self, S: jnp.ndarray) -> jnp.ndarray:
        """Full encode -> decode pipeline."""
        return self.decode(self.encode(S))

    def reconstruction_error(
        self, S: jnp.ndarray, *, relative: bool = True,
    ) -> float:
        """Frobenius reconstruction error."""
        S = jnp.asarray(S)
        err = float(jnp.linalg.norm(S - self.reconstruct(S), "fro"))
        if relative:
            return err / float(jnp.linalg.norm(S - self.s_ref_, "fro"))
        return err

    @property
    def n_active_nonlinear(self) -> int:
        """POD has no nonlinear features."""
        return 0


# ---------------------------------------------------------------------------
# Greedy-QM (Schwerdtner & Peherstorfer 2024)
# ---------------------------------------------------------------------------


class GreedyQM:
    """Greedy quadratic manifold with the full monomial shell active.

    At each iteration, each candidate mode j' from the s-pool is scored by
    the ridge-regularized least-squares fit of the projection residual
    (spanned by all non-selected modes) with quadratic monomial features of
    the trial coordinates; the candidate with the smallest fit residual is
    committed.  The final lifting matrix refits the full d-dimensional
    residual on the selected coordinates.

    Arguments
    ---------
    r : int
        Reduced dimension.
    s : int
        Candidate pool size (``s > r``).
    reg : float
        Tikhonov magnitude of the SVD filter ``sigma / (sigma^2 + reg^2)``.
    center : bool
        Whether to subtract the snapshot mean before the SVD.
    """

    def __init__(self, r: int, s: int, *, reg: float = 1e-13,
                 center: bool = True):
        if s <= r:
            raise ValueError(f"Need a candidate pool s > r; got s={s}, r={r}.")
        self.r = int(r)
        self.s = int(s)
        self.reg = float(reg)
        self.center = bool(center)

    def fit(self, S: jnp.ndarray) -> "GreedyQM":
        """Fit to snapshot data ``S`` of shape ``(d, n)``."""
        S = jnp.asarray(S)
        d, n = S.shape
        self.s_ref_ = (
            jnp.mean(S, axis=1, keepdims=True)
            if self.center else jnp.zeros((d, 1), dtype=S.dtype)
        )
        S_tilde = S - self.s_ref_

        U, sigma, Wt = jnp.linalg.svd(S_tilde, full_matrices=False)
        rank = int(sigma.shape[0])
        s_eff = min(self.s, rank)
        SigmaWt = sigma[:, None] * Wt                       # (rank, n)

        # -- greedy mode selection over the s-pool -------------------------
        J: list[int] = []
        pool = set(range(s_eff))

        score_batch = jax.vmap(_ridge_fit_error, in_axes=(0, 0, None))

        for _ in range(self.r):
            candidates = sorted(pool)
            cand_arr = np.asarray(candidates, dtype=np.int32)
            n_cand = len(candidates)

            # trial coordinates (n_cand, i+1, n)
            cand_rows = SigmaWt[cand_arr][:, None, :]
            if J:
                committed = jnp.broadcast_to(
                    SigmaWt[jnp.asarray(J)][None, :, :],
                    (n_cand, len(J), n),
                )
                trials = jnp.concatenate([committed, cand_rows], axis=1)
            else:
                trials = cand_rows

            # residual rows: all non-selected modes minus the candidate.
            # B_all stacks the full non-selected residual; the candidate's
            # own row is zeroed per candidate.
            out_all = np.setdiff1d(np.arange(rank), np.asarray(J))
            B_all = SigmaWt[jnp.asarray(out_all)]           # (K, n)
            pos = {int(m): k for k, m in enumerate(out_all)}
            B_batch = jnp.stack([
                B_all.at[pos[int(j)], :].set(0.0) for j in candidates
            ])                                              # (n_cand, K, n)

            P_batch = jax.vmap(quadratic_features)(trials)  # (n_cand, m, n)
            errors = score_batch(P_batch, B_batch, self.reg)
            j_star = candidates[int(jnp.argmin(errors))]
            J.append(j_star)
            pool.remove(j_star)

        # -- final basis and lifting on the exact residual -----------------
        self.J_ = np.asarray(J, dtype=np.int32)
        self.U_r_ = U[:, jnp.asarray(J)]
        S_hat = SigmaWt[jnp.asarray(J)]                     # (r, n)
        E = S_tilde - self.U_r_ @ S_hat

        P = quadratic_features(S_hat)
        Us, sv, Vt = jnp.linalg.svd(P, full_matrices=False)
        filt = sv / (sv ** 2 + self.reg ** 2)
        self.Vbar_ = ((E @ Vt.T) * filt[None, :]) @ Us.T    # (d, m)
        return self

    def encode(self, S: jnp.ndarray) -> jnp.ndarray:
        """Linear encoding ``s_hat = V^T (s - s_ref)``."""
        return self.U_r_.T @ (jnp.asarray(S) - self.s_ref_)

    def decode(self, S_hat: jnp.ndarray) -> jnp.ndarray:
        """Quadratic decoding with the full monomial shell."""
        S_hat = jnp.asarray(S_hat)
        return (
            self.s_ref_ + self.U_r_ @ S_hat
            + self.Vbar_ @ quadratic_features(S_hat)
        )

    def reconstruct(self, S: jnp.ndarray) -> jnp.ndarray:
        """Full encode -> decode pipeline."""
        return self.decode(self.encode(S))

    def reconstruction_error(
        self, S: jnp.ndarray, *, relative: bool = True,
    ) -> float:
        """Frobenius reconstruction error."""
        S = jnp.asarray(S)
        err = float(jnp.linalg.norm(S - self.reconstruct(S), "fro"))
        if relative:
            return err / float(jnp.linalg.norm(S - self.s_ref_, "fro"))
        return err

    @property
    def n_active_nonlinear(self) -> int:
        """The full quadratic shell is always active."""
        return self.r * (self.r + 1) // 2


# ---------------------------------------------------------------------------
# Geelen-QM (Geelen, Wright & Willcox 2023)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("n_iter",))
def _fista_group_lasso(
    G: jnp.ndarray,
    EPt: jnp.ndarray,
    lam: float,
    step: float,
    n_iter: int,
) -> jnp.ndarray:
    """FISTA for  min_V 1/2 tr(V G V^T) - tr(V EPt^T) + lam sum_k ||v_k||."""
    d, m = EPt.shape

    def prox(Z):
        col = jnp.linalg.norm(Z, axis=0)
        scale = jnp.maximum(0.0, 1.0 - (lam * step) / jnp.maximum(col, 1e-300))
        return Z * scale[None, :]

    def body(carry, _):
        V, Y, t = carry
        V_new = prox(Y - step * (Y @ G - EPt))
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        Y_new = V_new + ((t - 1.0) / t_new) * (V_new - V)
        return (V_new, Y_new, t_new), None

    V0 = jnp.zeros((d, m), dtype=EPt.dtype)
    (V, _, _), _ = jax.lax.scan(body, (V0, V0, 1.0), None, length=n_iter)
    return V


class GeelenQM:
    """Leading-r POD quadratic manifold with group-sparse column selection.

    The quadratic lifting matrix solves the group-sparse regularized
    least-squares problem via FISTA; whole monomial columns are driven to
    zero, and the surviving columns are refit by Tikhonov least squares to
    remove the shrinkage bias.

    Arguments
    ---------
    r : int
        Reduced dimension (leading-r POD modes; no greedy selection).
    gamma_sp_rel : float
        Group-lasso weight relative to the largest column correlation.
    gamma : float
        Tikhonov regularization of the debiased refit.
    n_iter : int
        FISTA iterations.
    active_tol : float
        Relative column-norm threshold defining the active set.
    center : bool
        Whether to subtract the snapshot mean before the SVD.
    """

    def __init__(self, r: int, *, gamma_sp_rel: float = 0.2,
                 gamma: float = 1e-13, n_iter: int = 400,
                 active_tol: float = 1e-8, center: bool = True):
        self.r = int(r)
        self.gamma_sp_rel = float(gamma_sp_rel)
        self.gamma = float(gamma)
        self.n_iter = int(n_iter)
        self.active_tol = float(active_tol)
        self.center = bool(center)

    def fit(self, S: jnp.ndarray) -> "GeelenQM":
        """Fit to snapshot data ``S`` of shape ``(d, n)``."""
        S = jnp.asarray(S)
        d, _ = S.shape
        self.s_ref_ = (
            jnp.mean(S, axis=1, keepdims=True)
            if self.center else jnp.zeros((d, 1), dtype=S.dtype)
        )
        St = S - self.s_ref_

        U, _, _ = jnp.linalg.svd(St, full_matrices=False)
        self.U_r_ = U[:, : self.r]
        S_hat = self.U_r_.T @ St
        E = St - self.U_r_ @ S_hat

        P = quadratic_features(S_hat)                       # (m, n)
        G = P @ P.T
        EPt = E @ P.T
        Lip = float(jnp.linalg.eigvalsh(G)[-1]) + 1e-30
        lam = self.gamma_sp_rel * float(jnp.max(jnp.linalg.norm(EPt, axis=0)))

        Vbar = _fista_group_lasso(G, EPt, lam, 1.0 / Lip, self.n_iter)
        col_norm = jnp.linalg.norm(Vbar, axis=0)
        active = jnp.where(col_norm > self.active_tol * jnp.max(col_norm))[0]
        if active.size == 0:                                # keep the strongest
            active = jnp.argmax(jnp.linalg.norm(EPt, axis=0))[None]
        self.active_ = np.asarray(active)

        # debiased Tikhonov refit of the selected columns
        P_a = P[active]
        Gram = P_a @ P_a.T + self.gamma * jnp.eye(active.size, dtype=S.dtype)
        self.Vbar_ = jnp.linalg.solve(Gram.T, (E @ P_a.T).T).T
        return self

    def encode(self, S: jnp.ndarray) -> jnp.ndarray:
        """Linear encoding ``s_hat = V^T (s - s_ref)``."""
        return self.U_r_.T @ (jnp.asarray(S) - self.s_ref_)

    def decode(self, S_hat: jnp.ndarray) -> jnp.ndarray:
        """Quadratic decoding with the active monomial columns."""
        S_hat = jnp.asarray(S_hat)
        P = quadratic_features(S_hat)[self.active_]
        return self.s_ref_ + self.U_r_ @ S_hat + self.Vbar_ @ P

    def reconstruct(self, S: jnp.ndarray) -> jnp.ndarray:
        """Full encode -> decode pipeline."""
        return self.decode(self.encode(S))

    def reconstruction_error(
        self, S: jnp.ndarray, *, relative: bool = True,
    ) -> float:
        """Frobenius reconstruction error."""
        S = jnp.asarray(S)
        err = float(jnp.linalg.norm(S - self.reconstruct(S), "fro"))
        if relative:
            return err / float(jnp.linalg.norm(S - self.s_ref_, "fro"))
        return err

    @property
    def n_active_nonlinear(self) -> int:
        """Number of surviving quadratic-monomial columns."""
        return int(self.active_.size)
