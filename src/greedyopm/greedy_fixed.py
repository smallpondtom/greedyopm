"""Greedy mode selection with a fixed, all-active orthonormal feature map.

:class:`GreedyFixedOPM` mirrors the greedy quadratic-manifold construction
of Schwerdtner & Peherstorfer (2024) but replaces the monomial features by
the empirically orthonormal polynomial features of the multivariate
Arnoldi recurrence.  No feature selection and no stability control: every
nonlinear feature stays active and the lifting is the plain least-squares
solution ``Vbar = (1/n) E P^T`` (exact by orthonormality).  This isolates
the effect of the orthonormal basis alone and serves as the unregularized
ablation of :class:`greedyopm.greedy.GreedyOPM`.
"""

from __future__ import annotations

import warnings
from functools import partial
from math import comb
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from .polynomials.arnoldi import (
    arnoldi_build,
    arnoldi_build_batch,
    arnoldi_eval,
)
from .polynomials.multindex import (
    graded_lex_multi_indices_for_degrees,
    normalize_degrees,
)


@partial(jax.jit, static_argnames=("nl_start",))
def _score_candidates_fixed(
    P_nl_all: jnp.ndarray,
    W_cand: jnp.ndarray,
    sigma_cands: jnp.ndarray,
    n: float,
    nl_start: int,
) -> jnp.ndarray:
    """All-active greedy scores ``sigma^2 + sum_k n ||r_k||^2``.

    The unregularized limit of the budget-calibrated score: with every
    feature active at lambda = 0, the polynomial gain of candidate c is
    ``n sum_k ||r_k||^2`` over the pool complement (all other candidates).

    Arguments
    ---------
    P_nl_all : (n_cand, m_nl, n)
    W_cand : (n, n_cand)
    sigma_cands : (n_cand,)
    n : float
    nl_start : int  (unused; kept for signature parity)

    Returns
    -------
    scores : (n_cand,)
    """
    w_P = jnp.einsum("nc,bmn->bcm", W_cand, P_nl_all)
    scaled = (sigma_cands / n)[None, :, None] * w_P
    sq = scaled ** 2
    norms_sq = jnp.sum(sq, axis=1) - jnp.diagonal(sq, axis1=0, axis2=1).T
    return sigma_cands ** 2 + n * jnp.sum(norms_sq, axis=1)


class GreedyFixedOPM:
    """Greedy mode selection, all nonlinear features active, no budget.

    Arguments
    ---------
    r : int
        Target reduced dimension.
    s : int
        Candidate pool size (``s > r``).
    p : int | list of int
        Polynomial degree specification (int means degrees 2, ..., p).
    n_reorth : int
        Additional Gram-Schmidt passes (effective count ``max(1, .)``).
    center : bool
        Whether to subtract the snapshot mean before the SVD.
    """

    def __init__(
        self,
        r: int,
        s: int,
        p: Union[int, Sequence[int]] = 2,
        *,
        n_reorth: int = 1,
        center: bool = True,
    ):
        if s <= r:
            raise ValueError(f"Need a candidate pool s > r; got s={s}, r={r}.")
        self.r = int(r)
        self.s = int(s)
        self.p = p
        self.degrees_ = normalize_degrees(p)
        self.n_reorth = int(n_reorth)
        self.center = bool(center)
        self._fitted = False

    # ----- degree capping ---------------------------------------------------

    @staticmethod
    def _effective_degrees(
        n_coords: int, n: int, degrees: Sequence[int],
    ) -> list:
        """Drop trailing degrees until the total feature count is <= n.

        With every feature active, more features than snapshots makes the
        empirical orthonormalization rank-deficient; the budget-calibrated
        model handles this by pruning, the fixed model by capping degrees.
        """
        eff = list(degrees)
        while eff:
            total = 1 + n_coords + sum(comb(n_coords + q - 1, q) for q in eff)
            if total <= n:
                return eff
            eff = eff[:-1]
        return eff

    # ----- fit --------------------------------------------------------------

    def fit(self, S: jnp.ndarray) -> "GreedyFixedOPM":
        """Fit to snapshot data ``S`` of shape ``(d, n)``."""
        S = jnp.asarray(S)
        d, n = S.shape
        n_f = float(n)
        n_reorth_eff = max(1, self.n_reorth)

        # --- 1. Center and rank-s truncated SVD ---------------------------
        if self.center:
            self.s_ref_ = jnp.mean(S, axis=1, keepdims=True)
        else:
            self.s_ref_ = jnp.zeros((d, 1), dtype=S.dtype)
        S_tilde = S - self.s_ref_

        U_full, sigma_full, Wt_full = jnp.linalg.svd(S_tilde, full_matrices=False)
        s_eff = min(self.s, int(sigma_full.shape[0]))
        U_s = U_full[:, :s_eff]
        sigma_s = sigma_full[:s_eff]
        W_s = Wt_full[:s_eff, :].T
        self.sigma_s_ = sigma_s
        SigmaWt = sigma_s[:, None] * W_s.T

        # --- 2. Greedy mode selection -------------------------------------
        J: list[int] = []
        J_complement = set(range(s_eff))

        for i in range(self.r):
            n_coords = i + 1
            deg_eff = self._effective_degrees(n_coords, n, self.degrees_)
            mi = graded_lex_multi_indices_for_degrees(n_coords, deg_eff)
            nl_start = 1 + n_coords

            candidates = sorted(J_complement)
            cand_arr = jnp.asarray(candidates, dtype=jnp.int32)
            n_cand = len(candidates)

            cand_rows = SigmaWt[cand_arr][:, None, :]
            if i == 0:
                trial_all = cand_rows
            else:
                committed = jnp.broadcast_to(
                    SigmaWt[jnp.asarray(J)][None, :, :], (n_cand, i, n),
                )
                trial_all = jnp.concatenate([committed, cand_rows], axis=1)

            Psi_all, _, _, _ = arnoldi_build_batch(
                trial_all, mi, n_reorth=n_reorth_eff,
            )
            scores = _score_candidates_fixed(
                Psi_all[:, nl_start:, :], W_s[:, cand_arr],
                sigma_s[cand_arr], n_f, nl_start,
            )
            j_star = candidates[int(jnp.argmax(scores))]
            J.append(j_star)
            J_complement.remove(j_star)

        # --- 3. Final basis and lifting matrix ----------------------------
        self.J_ = np.asarray(J, dtype=np.int32)
        self.U_r_ = U_s[:, jnp.asarray(J)]
        self.S_hat_ = SigmaWt[jnp.asarray(J)]

        deg_eff = self._effective_degrees(self.r, n, self.degrees_)
        self.degrees_eff_ = deg_eff
        if deg_eff != self.degrees_:
            warnings.warn(
                f"Requested degrees {self.degrees_} exceed the snapshot "
                f"count; using effective degrees {deg_eff}.",
                stacklevel=2,
            )
        mi_final = graded_lex_multi_indices_for_degrees(self.r, deg_eff)
        self._multi_indices_np_ = mi_final
        self.feature_degrees_ = mi_final.sum(axis=1).astype(np.int32)
        self.n_linear_ = 1 + self.r

        self.Psi_, self.arnoldi_basis_ = arnoldi_build(
            self.S_hat_, mi_final, n_reorth=n_reorth_eff,
        )

        P_nl = self.Psi_[self.n_linear_:]
        self.m_nl_ = int(P_nl.shape[0])
        E_final = S_tilde - self.U_r_ @ self.S_hat_
        self.Vbar_nl_ = (1.0 / n) * (E_final @ P_nl.T)      # exact LS lifting
        self.active_nl_global_ = jnp.arange(self.n_linear_, self.Psi_.shape[0])

        self._fitted = True
        return self

    # ----- encode / decode / reconstruct ------------------------------------

    def encode(self, S: jnp.ndarray) -> jnp.ndarray:
        """Linear encoding ``s_hat = V^T (s - s_ref)``."""
        S = jnp.asarray(S)
        squeeze = S.ndim == 1
        if squeeze:
            S = S[:, None]
        S_hat = self.U_r_.T @ (S - self.s_ref_)
        return S_hat.squeeze(axis=1) if squeeze else S_hat

    def decode(self, S_hat: jnp.ndarray) -> jnp.ndarray:
        """Decode with all nonlinear features active."""
        S_hat = jnp.asarray(S_hat)
        squeeze = S_hat.ndim == 1
        if squeeze:
            S_hat = S_hat[:, None]
        Psi_new = arnoldi_eval(S_hat, self.arnoldi_basis_)
        S_approx = (
            self.s_ref_
            + self.U_r_ @ S_hat
            + self.Vbar_nl_ @ Psi_new[self.active_nl_global_]
        )
        return S_approx.squeeze(axis=1) if squeeze else S_approx

    def reconstruct(self, S: jnp.ndarray) -> jnp.ndarray:
        """Full encode -> decode pipeline."""
        return self.decode(self.encode(S))

    # ----- Diagnostics ------------------------------------------------------

    def reconstruction_error(
        self, S: jnp.ndarray, *, relative: bool = True,
    ) -> float:
        """Frobenius reconstruction error of the full pipeline."""
        S = jnp.asarray(S)
        err = float(jnp.linalg.norm(S - self.reconstruct(S), "fro"))
        if relative:
            return err / float(jnp.linalg.norm(S - self.s_ref_, "fro"))
        return err

    @property
    def n_active_nonlinear(self) -> int:
        """Number of nonlinear features (all are active)."""
        return self.m_nl_

    @property
    def active_degree_counts(self) -> dict:
        """Map ``{degree: n_features}`` of the (all-active) nonlinear set."""
        degs = self.feature_degrees_[np.asarray(self.active_nl_global_)]
        degrees = getattr(self, "degrees_eff_", self.degrees_)
        return {int(q): int(np.count_nonzero(degs == q)) for q in degrees}

    @property
    def selected_modes(self) -> np.ndarray:
        """Indices of the greedily selected singular vectors."""
        return self.J_

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        modes = list(self.J_) if self._fitted else "?"
        return (
            f"GreedyFixedOPM(r={self.r}, s={self.s}, "
            f"degrees={self.degrees_}, modes={modes}, {status})"
        )
