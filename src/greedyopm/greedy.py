"""Greedy Orthogonal Polynomial Manifold under a certified Lipschitz budget.

Implements Algorithm `greedy-opm` of the paper.  The manifold ansatz is

    Gamma(s_hat) = s_ref + V s_hat + Vbar p(s_hat) ,

where ``p`` collects the degree->=2 empirically orthonormal polynomial
features built by the multivariate Vandermonde-with-Arnoldi recurrence, and
the lifting matrix ``Vbar`` solves the Lipschitz budget Lagrangian exactly:
by orthonormality the fit separates over feature columns, one group
soft-threshold step is the global minimizer, and the multiplier ``lambda*``
enforcing the user's budget ``L`` is calibrated in closed form by a single
sort (see :mod:`greedyopm.budget`).

Greediness enters in exactly one place, the choice of the ``r`` POD modes.
At each iteration every candidate mode j' from the s-pool is scored by the
exact training-error reduction of its own budget-calibrated manifold,

    score(j') = sigma_{j'}^2 + sum_{k active} G_k^{(j')}(lambda^{(j')}) ,

with a fresh Arnoldi basis, a gradient pass on the delta-dilated
coordinates for the sup-gradient bounds eta_k, and its own calibrated
multiplier -- so all candidates are compared at a common certificate level.
Candidate evaluation is vectorised with ``vmap``; the scoring runs entirely
in the SVD-reduced space, so the state dimension ``d`` enters only through
the initial SVD and the final lifting assembly.

The returned decoder carries the certificate

    L_hat_Gamma(delta) = sqrt(1 + (sum_k eta_k ||vbar_k||)^2) <= L ,

which combined with the OOD theorem yields the explicit guarantee
``||s - Gamma(E(s))|| <= min_l [(1 + L) ||s - s^(l)|| + eps_l]``.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from .budget import (
    calibrate_multiplier,
    certificate_from_budget,
    feature_gains,
    shrink_factors,
)
from .polynomials.arnoldi import (
    ArnoldiBasis,
    arnoldi_build,
    arnoldi_build_batch,
    arnoldi_eval,
    arnoldi_eval_grad,
    arnoldi_grad_batch,
)
from .polynomials.multindex import (
    degree_slices_for_degrees,
    graded_lex_multi_indices_for_degrees,
    normalize_degrees,
)


# ---------------------------------------------------------------------------
# JIT-compiled candidate scoring (one greedy iteration, all candidates)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("nl_start",))
def _score_candidates(
    P_nl_all: jnp.ndarray,
    G_all: jnp.ndarray,
    W_cand: jnp.ndarray,
    sigma_cands: jnp.ndarray,
    B_L: jnp.ndarray,
    n: float,
    nl_start: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Budget-calibrated greedy scores for all candidate modes at once.

    For candidate c the trial basis is J u {j_c}; its residual correlations
    are evaluated in the SVD-reduced space over the pool complement
    (all other candidates), its sup-gradient bounds eta_k come from the
    gradient pass on the dilated probe coordinates, and its multiplier is
    calibrated to the budget before scoring (Proposition `mode_decomp`).

    Arguments
    ---------
    P_nl_all : (n_cand, m_nl, n)
        Nonlinear feature blocks of the per-candidate Arnoldi bases.
    G_all : (n_cand, M, n_coords, n_probe)
        Per-candidate feature gradients on the dilated probe points.
    W_cand : (n, n_cand)
        Right-singular-vector columns of the candidate modes.
    sigma_cands : (n_cand,)
        Singular values of the candidate modes.
    B_L : scalar
        Budget bound sqrt(L^2 - 1).
    n : float
        Number of training snapshots.
    nl_start : int
        Row index where the nonlinear features start (1 + n_coords).

    Returns
    -------
    scores : (n_cand,)   sigma^2 + calibrated polynomial gain.
    poly_gain : (n_cand,)
    n_active : (n_cand,)  active-feature counts at each candidate's lambda.
    """
    # -- residual correlation norms in the reduced space ---------------------
    # w_P[c, c', k] = W_s[:, cand[c']]^T . P_nl_all[c, k, :];  the residual
    # complement of candidate c is all other candidates, so the self term
    # (diagonal c' == c) is excluded from the norm.
    w_P = jnp.einsum("nc,bmn->bcm", W_cand, P_nl_all)
    scaled = (sigma_cands / n)[None, :, None] * w_P     # (b, c, m)
    sq = scaled ** 2
    norms_sq = jnp.sum(sq, axis=1) - jnp.diagonal(sq, axis1=0, axis2=1).T
    r_norms = jnp.sqrt(jnp.maximum(norms_sq, 0.0))      # (b, m)

    # -- sup-gradient bounds eta_k on the dilated probe cloud ----------------
    G_nl = G_all[:, nl_start:, :, :]                    # (b, m, n_coords, n_probe)
    eta = jnp.sqrt(jnp.max(jnp.sum(G_nl ** 2, axis=2), axis=2))

    # -- per-candidate calibration and additive gains ------------------------
    lam = jax.vmap(calibrate_multiplier, in_axes=(0, 0, None, None))(
        r_norms, eta, B_L, n,
    )                                                   # (b,)
    gains = jax.vmap(feature_gains, in_axes=(0, 0, 0, None))(
        r_norms, eta, lam, n,
    )                                                   # (b, m)
    poly_gain = jnp.sum(gains, axis=1)
    n_active = jnp.sum(gains > 0.0, axis=1)
    return sigma_cands ** 2 + poly_gain, poly_gain, n_active


# ---------------------------------------------------------------------------
# GreedyOPM
# ---------------------------------------------------------------------------


class GreedyOPM:
    """Greedy Orthogonal Polynomial Manifold under a certified Lipschitz budget.

    Arguments
    ---------
    r : int
        Target reduced dimension (number of POD modes to select).
    s : int
        Candidate pool size (``s > r``): the greedy selection picks modes
        from the first ``s`` left-singular vectors.
    p : int | list of int
        Polynomial degree specification.

        * int: use all nonlinear degrees 2, 3, ..., p.
        * list: use exactly those nonlinear degrees (each >= 2).
    L : float
        Lipschitz budget ``L > 1``.  The returned decoder satisfies the
        certificate ``L_hat_Gamma(delta) <= L``, so ``L`` is the slope
        constant of the OOD guarantee.  ``L = inf`` disables the budget
        (unregularized least-squares lifting, all features active).
    delta : float
        Extrapolation margin ``delta >= 0``: the sup-gradient bounds eta_k
        are evaluated on the dilated training coordinates
        ``(1 + delta) s_hat^(l)``, declaring how far beyond the training
        data the certificate must hold.
    n_reorth : int
        Number of additional Gram-Schmidt re-orthogonalisation passes
        beyond the first ("twice is enough" is load-bearing in finite
        precision; the effective count is ``max(1, n_reorth)``).
    n_grad_samples : int, optional
        Number of evenly spaced training snapshots used as probe points
        for the gradient passes.  ``None`` (default) uses all ``n``.
        Subsampling weakens the certificate to the probed subset.
    center : bool
        Whether to subtract the snapshot mean before the SVD.

    Attributes (after :meth:`fit`)
    ------------------------------
    J_ : (r,) selected mode indices into the s-pool.
    U_r_ : (d, r) selected POD basis V.
    S_hat_ : (r, n) training reduced coordinates.
    Psi_ : (M, n) orthonormal features on the training coordinates.
    arnoldi_basis_ : stored recurrence data for out-of-sample evaluation.
    C_nl_ : (d, m_nl) residual correlations r_k (columns).
    r_norms_ : (m_nl,) their norms ||r_k||.
    eta_ : (m_nl,) sup-gradient bounds on the dilated probe cloud.
    lambda_ : calibrated multiplier lambda*.
    Vbar_nl_ : (d, |A|) active lifting columns (soft-thresholded).
    active_nl_global_ : row indices of the active features in ``Psi_``.
    budget_ : attained budget B = sum_k eta_k ||vbar_k||.
    certificate_ : L_hat_Gamma(delta) = sqrt(1 + B^2)  (<= L).
    """

    def __init__(
        self,
        r: int,
        s: int,
        p: Union[int, Sequence[int]] = 2,
        *,
        L: float = 10.0,
        delta: float = 0.1,
        n_reorth: int = 1,
        n_grad_samples: Optional[int] = None,
        center: bool = True,
    ):
        if not (L > 1.0):
            raise ValueError(f"The Lipschitz budget must satisfy L > 1; got {L}.")
        if delta < 0.0:
            raise ValueError(f"The dilation must satisfy delta >= 0; got {delta}.")
        if s <= r:
            raise ValueError(f"Need a candidate pool s > r; got s={s}, r={r}.")

        self.r = int(r)
        self.s = int(s)
        self.p = p
        self.degrees_ = normalize_degrees(p)
        self.L = float(L)
        self.delta = float(delta)
        self.n_reorth = int(n_reorth)
        self.n_grad_samples = n_grad_samples
        self.center = bool(center)
        self._fitted = False

    # =====================================================================
    # fit
    # =====================================================================

    def fit(self, S: jnp.ndarray) -> "GreedyOPM":
        """Fit the manifold to snapshot data ``S`` of shape ``(d, n)``.

        Runs Algorithm `greedy-opm`: r greedy iterations, each rebuilding a
        fresh Arnoldi basis per candidate (vectorised over candidates), a
        gradient pass on the dilated coordinates, per-candidate closed-form
        budget calibration, and exact additive scoring.  After the loop the
        final basis, exact d-dimensional residual correlations, and final
        calibration produce the lifting matrix and its certificate.
        """
        S = jnp.asarray(S)
        d, n = S.shape
        n_f = float(n)
        n_reorth_eff = max(1, self.n_reorth)
        B_L = jnp.asarray(np.sqrt(self.L ** 2 - 1.0), dtype=S.dtype)

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
        W_s = Wt_full[:s_eff, :].T                          # (n, s_eff)
        self.sigma_s_ = sigma_s
        SigmaWt = sigma_s[:, None] * W_s.T                  # (s_eff, n)

        probe_idx = self._probe_indices(n)
        dilate = 1.0 + self.delta

        # --- 2. Greedy mode selection -------------------------------------
        J: list[int] = []
        J_complement = set(range(s_eff))
        self.iteration_records_: list[dict] = []

        for i in range(self.r):
            n_coords = i + 1
            candidates = sorted(J_complement)
            cand_arr = jnp.asarray(candidates, dtype=jnp.int32)
            n_cand = len(candidates)

            # -- 2a. Trial coordinates for every candidate ----------------
            # Rows of trial c are [S_hat(J), SigmaWt[cand[c]]].
            cand_rows = SigmaWt[cand_arr][:, None, :]       # (n_cand, 1, n)
            if i == 0:
                trial_all = cand_rows
            else:
                committed = jnp.broadcast_to(
                    SigmaWt[jnp.asarray(J)][None, :, :], (n_cand, i, n),
                )
                trial_all = jnp.concatenate([committed, cand_rows], axis=1)

            # -- 2b. Fresh Arnoldi basis per candidate (vmapped) ----------
            mi = graded_lex_multi_indices_for_degrees(n_coords, self.degrees_)
            Psi_all, H_all, ell_arr, parent_arr = arnoldi_build_batch(
                trial_all, mi, n_reorth=n_reorth_eff,
            )                                               # (n_cand, M, n)
            nl_start = 1 + n_coords

            # -- 2c. Gradient pass on the dilated probe coordinates -------
            probes = dilate * trial_all[:, :, probe_idx]
            _, G_all = arnoldi_grad_batch(probes, H_all, ell_arr, parent_arr)

            # -- 2d. Calibrate every candidate and score ------------------
            scores, poly_gain, n_active = _score_candidates(
                Psi_all[:, nl_start:, :], G_all, W_s[:, cand_arr],
                sigma_s[cand_arr], B_L, n_f, nl_start,
            )

            # -- 2e. Commit the winner ------------------------------------
            best = int(jnp.argmax(scores))
            j_star = candidates[best]
            J.append(j_star)
            J_complement.remove(j_star)
            self.iteration_records_.append({
                "iteration": i,
                "selected_mode": j_star,
                "score": float(scores[best]),
                "linear_gain": float(sigma_s[j_star] ** 2),
                "poly_gain": float(poly_gain[best]),
                "n_active_features": int(n_active[best]),
            })

        # --- 3. Final basis on the selected modes -------------------------
        self.J_ = np.asarray(J, dtype=np.int32)
        self.U_r_ = U_s[:, jnp.asarray(J)]                  # (d, r)
        self.S_hat_ = SigmaWt[jnp.asarray(J)]               # (r, n)

        mi_final = graded_lex_multi_indices_for_degrees(self.r, self.degrees_)
        self._multi_indices_np_ = mi_final
        self.feature_degrees_ = mi_final.sum(axis=1).astype(np.int32)
        self.n_linear_ = 1 + self.r

        self.Psi_, self.arnoldi_basis_ = arnoldi_build(
            self.S_hat_, mi_final, n_reorth=n_reorth_eff,
        )

        # --- 4. Exact residual correlations and sup-gradient bounds -------
        P_nl = self.Psi_[self.n_linear_:]                   # (m_nl, n)
        self.m_nl_ = int(P_nl.shape[0])

        E_final = S_tilde - self.U_r_ @ self.S_hat_         # (d, n)
        self.C_nl_ = (1.0 / n) * (E_final @ P_nl.T)         # (d, m_nl): r_k
        self.r_norms_ = jnp.linalg.norm(self.C_nl_, axis=0)

        probes = dilate * self.S_hat_[:, probe_idx]
        _, G_final = arnoldi_eval_grad(probes, self.arnoldi_basis_)
        G_nl = G_final[self.n_linear_:]                     # (m_nl, r, n_probe)
        self.eta_ = jnp.sqrt(jnp.max(jnp.sum(G_nl ** 2, axis=1), axis=1))

        # --- 5. Final calibration, soft-threshold, certificate ------------
        self.set_budget(self.L)
        self._fitted = True
        return self

    # =====================================================================
    # Budget re-calibration (cheap L sweep -- no refit of the mode selection)
    # =====================================================================

    def set_budget(self, L: float) -> "GreedyOPM":
        """Re-calibrate the lifting matrix for a new Lipschitz budget ``L``.

        Reuses the stored residual correlations ``C_nl_`` and sup-gradient
        bounds ``eta_``, so sweeping ``L`` costs one sort plus one column
        rescaling -- no SVD, greedy selection, or Arnoldi rebuild.  Note the
        greedy *mode* selection is kept from the fit-time budget; refit for
        a fully L-consistent model.
        """
        if not (L > 1.0):
            raise ValueError(f"The Lipschitz budget must satisfy L > 1; got {L}.")
        self.L = float(L)
        n = float(self.Psi_.shape[1])
        B_L = jnp.asarray(np.sqrt(self.L ** 2 - 1.0), dtype=self.r_norms_.dtype)

        lam = calibrate_multiplier(self.r_norms_, self.eta_, B_L, n)
        shrink = shrink_factors(self.r_norms_, self.eta_, lam, n)

        active_local = jnp.where(shrink > 0.0)[0]
        self.lambda_ = float(lam)
        self.active_nl_global_ = self.n_linear_ + active_local
        self.Vbar_nl_ = self.C_nl_[:, active_local] * shrink[active_local][None, :]

        vbar_norms = self.r_norms_[active_local] * shrink[active_local]
        self.budget_ = float(jnp.sum(self.eta_[active_local] * vbar_norms))
        self.certificate_ = float(certificate_from_budget(jnp.asarray(self.budget_)))
        return self

    # =====================================================================
    # encode / decode / reconstruct
    # =====================================================================

    def encode(self, S: jnp.ndarray) -> jnp.ndarray:
        """Linear encoding ``s_hat = V^T (s - s_ref)``."""
        S = jnp.asarray(S)
        squeeze = S.ndim == 1
        if squeeze:
            S = S[:, None]
        S_hat = self.U_r_.T @ (S - self.s_ref_)
        return S_hat.squeeze(axis=1) if squeeze else S_hat

    def decode(self, S_hat: jnp.ndarray) -> jnp.ndarray:
        """Decode ``Gamma(s_hat) = s_ref + V s_hat + Vbar p(s_hat)``.

        Features at the new coordinates are evaluated by the stored Arnoldi
        recurrence; only the active nonlinear rows enter the lifting.
        """
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

    # =====================================================================
    # Diagnostics
    # =====================================================================

    def reconstruction_error(
        self, S: jnp.ndarray, *, relative: bool = True,
    ) -> float:
        """Frobenius reconstruction error of the full pipeline."""
        S = jnp.asarray(S)
        err = float(jnp.linalg.norm(S - self.reconstruct(S), "fro"))
        if relative:
            return err / float(jnp.linalg.norm(S - self.s_ref_, "fro"))
        return err

    def linear_error(self, S: jnp.ndarray, *, relative: bool = True) -> float:
        """Frobenius error of the linear (POD-only) part of the manifold."""
        S = jnp.asarray(S)
        S_centered = S - self.s_ref_
        S_lin = self.s_ref_ + self.U_r_ @ (self.U_r_.T @ S_centered)
        err = float(jnp.linalg.norm(S - S_lin, "fro"))
        if relative:
            return err / float(jnp.linalg.norm(S_centered, "fro"))
        return err

    def lipschitz_certificate(
        self,
        *,
        dilation: Optional[float] = None,
        S_hat: Optional[jnp.ndarray] = None,
    ) -> float:
        """A-posteriori certificate on an arbitrary probe set.

        Recomputes ``sqrt(1 + (sum_k ||vbar_k|| L_hat_psi_k)^2)`` with the
        per-feature sup-gradients taken over the given probe coordinates
        (default: the training coordinates dilated by ``dilation``).  With
        the fit-time probes this returns ``certificate_``.
        """
        if S_hat is None:
            d_eff = self.delta if dilation is None else float(dilation)
            S_hat = (1.0 + d_eff) * self.S_hat_
        _, G = arnoldi_eval_grad(S_hat, self.arnoldi_basis_)
        G_act = G[self.active_nl_global_]                   # (|A|, r, n_pts)
        L_psi = jnp.sqrt(jnp.max(jnp.sum(G_act ** 2, axis=1), axis=1))
        v_norms = jnp.linalg.norm(self.Vbar_nl_, axis=0)
        return float(certificate_from_budget(jnp.sum(v_norms * L_psi)))

    def decoder_jacobian_norms(self, S_hat: jnp.ndarray) -> jnp.ndarray:
        """Pointwise spectral norms ``||J_Gamma(s_hat)||_2``.

        Uses the splitting ``||J_Gamma||^2 = 1 + ||Vbar J_p||^2`` (Lemma
        `split`), evaluating the second term through the Gram matrix
        ``Vbar^T Vbar`` so no d-dimensional object appears per point.
        """
        _, G = arnoldi_eval_grad(jnp.asarray(S_hat), self.arnoldi_basis_)
        Jp = G[self.active_nl_global_]                      # (|A|, r, n_pts)
        M_gram = self.Vbar_nl_.T @ self.Vbar_nl_

        def one_point(Jp_pt):
            A = Jp_pt.T @ (M_gram @ Jp_pt)
            lam_max = jnp.linalg.eigvalsh(A)[-1]
            return jnp.sqrt(1.0 + jnp.maximum(lam_max, 0.0))

        return jax.vmap(one_point, in_axes=2)(Jp)

    @property
    def n_active_nonlinear(self) -> int:
        """Number of active (post-threshold) nonlinear features."""
        return int(self.active_nl_global_.shape[0])

    @property
    def active_degree_counts(self) -> dict:
        """Map ``{degree: n_active_features}`` for the active nonlinear set."""
        active = np.asarray(self.active_nl_global_)
        degs = self.feature_degrees_[active]
        return {int(q): int(np.count_nonzero(degs == q)) for q in self.degrees_}

    @property
    def selected_modes(self) -> np.ndarray:
        """Indices of the greedily selected singular vectors."""
        return self.J_

    def summary(self) -> dict:
        """Configuration and fit results as a plain dict."""
        return {
            "r": self.r,
            "s": self.s,
            "degrees": self.degrees_,
            "L": self.L,
            "delta": self.delta,
            "selected_modes": [int(j) for j in self.J_],
            "n_snapshots": int(self.Psi_.shape[1]),
            "state_dim": int(self.U_r_.shape[0]),
            "total_nonlinear_features": self.m_nl_,
            "active_nonlinear_features": self.n_active_nonlinear,
            "active_by_degree": self.active_degree_counts,
            "lambda": self.lambda_,
            "budget": self.budget_,
            "certificate": self.certificate_,
        }

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        modes = list(self.J_) if self._fitted else "?"
        return (
            f"GreedyOPM(r={self.r}, s={self.s}, degrees={self.degrees_}, "
            f"L={self.L}, delta={self.delta}, modes={modes}, {status})"
        )

    # ----- helpers ----------------------------------------------------------

    def _probe_indices(self, n: int) -> np.ndarray:
        """Evenly spaced snapshot indices used as gradient probe points."""
        if self.n_grad_samples is None or int(self.n_grad_samples) >= n:
            return np.arange(n)
        return np.unique(
            np.linspace(0, n - 1, int(self.n_grad_samples)).astype(np.int64)
        )
