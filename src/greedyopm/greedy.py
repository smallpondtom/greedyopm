"""Greedy Incremental Orthogonal Polynomial Manifold (Algorithm 1 of the paper).

This module implements the two greedy variants described in the paper:

* :class:`GreedyOPM` -- joint greedy selection of POD modes AND polynomial
  features via group soft-thresholding.  At each iteration, the full
  multivariate Arnoldi basis is rebuilt from scratch for every candidate
  j', with the rebuild vectorised over candidates via ``vmap``.  This gives
  the "canonical" orthonormal polynomial basis (in grlex order) at every
  step, so the NL subspace is the true orthogonal complement of
  ``span{1, ŝ_1, ..., ŝ_i}`` within the polynomial space.  The paper's
  per-iteration incremental Arnoldi (Section 5.5) is available separately
  in :mod:`greedyopm.polynomials.incremental_arnoldi` for experimentation,
  but is NOT used here: in practice it produces a tilted NL subspace
  (new linear rows get orthogonalised against old NL rows), which degrades
  mode selection on problems with slow Kolmogorov-n-width decay.

* :class:`GreedyFixedOPM` -- greedy mode selection with all nonlinear
  features kept active (a la Schwerdtner & Peherstorfer 2024) but using
  empirically orthonormal polynomial features.

Both classes expose the same encode / decode / reconstruct API and accept
``ortho_method in {"qr", "cholesky", "arnoldi"}``.  For :class:`GreedyOPM`
``ortho_method`` affects the post-selection basis used for out-of-sample
evaluation; selection itself uses Arnoldi regardless.

Memory / compute optimisations
------------------------------
* Per-iteration per-candidate Arnoldi rebuilds are fully vectorised via
  :func:`arnoldi_build_batch`.
* Candidate scoring uses the SVD-compressed projections
  ``w_P_nl = W_s^T P_nl^T`` of shape ``(s, m_nl)``, formed with an einsum
  over the batch.  The ``(d, n)``-wide snapshot / feature matrix is never
  touched inside the inner loop.
"""

from __future__ import annotations

import functools
import warnings
from typing import Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from .polynomials.arnoldi import (
    ArnoldiBasis,
    arnoldi_build,
    arnoldi_build_batch,
    arnoldi_eval,
)
from .polynomials.classical_tensor import (
    _FAMILY_DISPATCH,
    evaluate_classical_tensor,
)
from .polynomials.monomial import evaluate_monomials
from .polynomials.multindex import (
    _normalize_degrees,
    degree_slices_for_degrees,
    graded_lex_multi_indices_for_degrees,
)
from .polynomials.orthogonalize import (
    _safe_inverse_factor_cholesky,
    _safe_inverse_factor_qr,
    orthogonalize_cholesky,
    orthogonalize_cholesky_refined,
    orthogonalize_qr,
    orthogonalize_qr_refined,
)


_ORTHO_METHODS = ("qr", "cholesky", "arnoldi")


def _validate_ortho_family(
    ortho_method: str, family: str, cls_name: str,
) -> None:
    """Warn when Arnoldi is combined with a non-monomial classical family."""
    if ortho_method == "arnoldi" and family != "monomial":
        warnings.warn(
            f"{cls_name}: ortho_method='arnoldi' ignores family='{family}'. "
            "Arnoldi builds orthogonal polynomials directly from the raw "
            "reduced coordinates.",
            stacklevel=3,
        )


# =============================================================================
# GreedyOPM  --  joint mode + feature selection (Algorithm 1)
# =============================================================================


class GreedyOPM:
    """Greedy Incremental Orthogonal Polynomial Manifold.

    Parameters
    ----------
    r : int
        Target reduced dimension (number of POD modes to select).
    s : int
        Candidate pool size (``s > r``).
    p : int | list of int
        Polynomial degree specification.

        * int: use all nonlinear degrees 2, 3, ..., p.
        * list: use exactly those polynomial degrees (each >= 2).
    gamma_sp : float
        Sparsity parameter for group soft-thresholding.  A feature k is
        active iff ``||r_tilde_k||_2 > gamma_sp / n``, equivalently iff the
        error reduction from activating it exceeds ``gamma_sp^2 / n``.
    gamma : float
        Tikhonov regularisation for the final lifting matrix.
    n_reorth : int
        Number of additional Gram-Schmidt re-orthogonalisation passes beyond
        the first.  The Arnoldi paths enforce ``max(1, n_reorth)`` since the
        "twice is enough" principle is load-bearing in finite precision.
    R_rank_tol : float
        Rank-truncation tolerance for ``R^{-T}`` (QR / Cholesky post-rebuild
        paths only).  Ignored when ``ortho_method='arnoldi'``.
    family : {"monomial", "legendre", "chebyshev", "hermite"}
        Starting polynomial family (QR / Cholesky paths only).
    basis : {"hybrid", "hierarchical"}
        "hybrid" = classical + empirical QR; "hierarchical" = monomial.
        Ignored when ``ortho_method='arnoldi'``.
    ortho_method : {"qr", "cholesky", "arnoldi"}
        Post-selection basis construction.  Affects only the out-of-sample
        decode path.  The greedy mode selection is basis-invariant within a
        fixed span, so the selected modes are independent of this choice.
    center : bool
        Whether to subtract the snapshot mean before SVD.
    """

    def __init__(
        self,
        r: int,
        s: int,
        p: Union[int, Sequence[int]] = 2,
        *,
        gamma_sp: float = 0.0,
        gamma: float = 1e-8,
        n_reorth: int = 1,
        R_rank_tol: float = 0.0,
        family: Literal["monomial", "legendre", "chebyshev", "hermite"] = "monomial",
        basis: Literal["hybrid", "hierarchical"] = "hybrid",
        ortho_method: Literal["qr", "cholesky", "arnoldi"] = "arnoldi",
        center: bool = True,
    ):
        if ortho_method not in _ORTHO_METHODS:
            raise ValueError(
                f"ortho_method must be one of {_ORTHO_METHODS}; got {ortho_method!r}."
            )
        _validate_ortho_family(ortho_method, family, "GreedyOPM")

        self.r = r
        self.s = s
        self.p = p
        self.degrees_ = _normalize_degrees(p)
        self.p_max_ = max(self.degrees_)
        self.gamma_sp = gamma_sp
        self.gamma = gamma
        self.n_reorth = n_reorth
        self.R_rank_tol = R_rank_tol
        self.family = family
        self.basis = basis
        self.ortho_method = ortho_method
        self.center = center
        self._fitted = False

    # =====================================================================
    # fit
    # =====================================================================

    def fit(self, S: jnp.ndarray) -> "GreedyOPM":
        """Fit the greedy OPM to snapshot data ``S`` of shape ``(d, n)``.

        At each iteration i (0-indexed), we rebuild the full orthonormal
        polynomial basis from scratch for each candidate j' by running the
        multivariate Arnoldi process on the (i+1)-dimensional reduced
        coordinates (J_{i-1} ∪ {j'}).  This is vectorised across candidates
        via ``vmap``.

        Rationale: the paper's incremental Arnoldi scheme (Section 5.5)
        produces a polynomial basis that differs from the "canonical" grlex
        basis: the new linear row at iteration i is orthogonalised against
        old NONlinear rows, which tilts the linear subspace away from
        ``span{1, ŝ_1, ..., ŝ_i}`` and correspondingly tilts the NL
        subspace.  Empirically this causes the greedy mode selection to
        degrade significantly for problems with slow Kolmogorov n-width
        decay (e.g., linear transport).  The full per-iteration rebuild is
        a factor of 2-3x slower in wall time but produces the "correct"
        NL subspace at every step, matching :class:`GreedyFixedOPM` quality
        while also admitting group soft-thresholding.
        """
        d, n = S.shape
        n_f = jnp.asarray(float(n), dtype=S.dtype)
        gamma_sp_f = jnp.asarray(float(self.gamma_sp), dtype=S.dtype)

        # --- 1. Center ----------------------------------------------------
        if self.center:
            self.s_ref_ = jnp.mean(S, axis=1, keepdims=True)
        else:
            self.s_ref_ = jnp.zeros((d, 1), dtype=S.dtype)
        S_tilde = S - self.s_ref_

        # --- 2. Rank-s truncated SVD --------------------------------------
        U_full, sigma_full, Wt_full = jnp.linalg.svd(
            S_tilde, full_matrices=False,
        )
        s_eff = min(self.s, int(len(sigma_full)))
        U_s = U_full[:, :s_eff]
        sigma_s = sigma_full[:s_eff]
        W_s = Wt_full[:s_eff, :].T                                    # (n, s_eff)
        self.sigma_s_ = sigma_s
        SigmaWt = sigma_s[:, None] * W_s.T                            # (s_eff, n)

        n_reorth_eff = max(1, int(self.n_reorth))

        # --- 3. Greedy loop -----------------------------------------------
        J: list[int] = []
        J_complement: set[int] = set(range(s_eff))
        self._iteration_records: list[dict] = []

        for i in range(self.r):
            n_coords_trial = i + 1
            candidates_py = sorted(J_complement)
            candidates_arr = jnp.array(candidates_py, dtype=jnp.int32)
            n_cand = int(candidates_arr.shape[0])

            # -- 3a. Build trial S_hat for each candidate -----------------
            # Shape: (n_cand, n_coords_trial, n).  For each candidate c,
            # rows are [SigmaWt[J[0]], ..., SigmaWt[J[i-1]], SigmaWt[cand[c]]].
            cand_row = SigmaWt[candidates_arr][:, None, :]            # (n_cand, 1, n)
            if i == 0:
                trial_S_hat_all = cand_row
            else:
                S_hat_old = SigmaWt[jnp.asarray(J)]                    # (i, n)
                S_hat_old_b = jnp.broadcast_to(
                    S_hat_old[None, :, :], (n_cand, i, n),
                )
                trial_S_hat_all = jnp.concatenate(
                    [S_hat_old_b, cand_row], axis=1,
                )                                                      # (n_cand, i+1, n)

            # -- 3b. Shared grlex multi-indices ---------------------------
            all_mi_np = graded_lex_multi_indices_for_degrees(
                n_coords_trial, self.degrees_,
            )
            all_mi_np = np.asarray(all_mi_np, dtype=np.int32)

            # -- 3c. Batched full Arnoldi over candidates -----------------
            Psi_all, _, _, _ = arnoldi_build_batch(
                trial_S_hat_all, all_mi_np, n_reorth=n_reorth_eff,
            )                                                          # (n_cand, M_i, n)

            # -- 3d. NL block projections in the SVD-reduced space --------
            nl_start = 1 + n_coords_trial   # grlex: constant + linears then NL
            P_nl_all = Psi_all[:, nl_start:, :]                       # (n_cand, m_nl, n)
            # w_P_nl[c, l, k] = W_s[:, l]^T · P_nl_all[c, k, :]
            w_P_nl_all = jnp.einsum(
                "nl,cmn->clm", W_s, P_nl_all,
            )                                                          # (n_cand, s, m_nl)

            # -- 3e. Score each candidate ---------------------------------
            sigma_cands = sigma_s[candidates_arr]                      # (n_cand,)
            # J_c = candidates_py \ {candidates_py[c]}.  For the sum over
            # J_c, broadcast across the candidate axis and exclude the
            # self-index.
            w_P_at_cands = w_P_nl_all[:, candidates_arr, :]            # (n_cand, n_cand, m_nl)
            inv_n = 1.0 / n_f
            scaled = (inv_n * sigma_cands)[None, :, None] * w_P_at_cands
            sq = scaled ** 2                                           # (n_cand, n_cand, m_nl)
            total = jnp.sum(sq, axis=1)                                # (n_cand, m_nl)
            diag_ = jnp.diagonal(sq, axis1=0, axis2=1).T               # (n_cand, m_nl)
            norms_sq = total - diag_                                   # per-cand, per-feature

            threshold_sq = (gamma_sp_f * inv_n) ** 2
            active_mask = norms_sq > threshold_sq
            poly_gain = n_f * jnp.sum(norms_sq * active_mask, axis=1)  # (n_cand,)
            delta = sigma_cands ** 2 + poly_gain                       # absolute scoring

            # -- 3f. Pick winner ------------------------------------------
            best_local = int(jnp.argmax(delta))
            j_star = candidates_py[best_local]

            J.append(j_star)
            J_complement.remove(j_star)

            self._iteration_records.append({
                "iteration": i,
                "selected_mode": j_star,
                "delta": float(delta[best_local]),
                "linear_gain": float(sigma_s[j_star] ** 2),
                "poly_gain": float(poly_gain[best_local]),
                "n_active_features": int(jnp.sum(active_mask[best_local])),
            })

        # --- 4. Finalise: selected modes + final basis --------------------
        self.J_ = np.asarray(J, dtype=np.int32)
        self.U_r_ = U_s[:, jnp.asarray(J)]                             # (d, r)
        self.S_hat_ = SigmaWt[jnp.asarray(J)]                          # (r, n)

        all_mi_final_np = graded_lex_multi_indices_for_degrees(
            self.r, self.degrees_,
        )
        all_mi_final_np = np.asarray(all_mi_final_np, dtype=np.int32)
        self.multi_indices_ = jnp.asarray(all_mi_final_np)
        self._multi_indices_np_ = all_mi_final_np

        slices = degree_slices_for_degrees(self.r, self.degrees_)
        self.n_linear_ = slices[1].stop if 1 in slices else slices[0].stop

        # Build final Psi according to ortho_method.
        self.arnoldi_basis_: Optional[ArnoldiBasis] = None
        self.RL_inv_T_: Optional[jnp.ndarray] = None

        if self.ortho_method == "arnoldi":
            self.Psi_, self.arnoldi_basis_ = arnoldi_build(
                self.S_hat_, all_mi_final_np, n_reorth=n_reorth_eff,
            )
            self.eff_rank_ = int(self.Psi_.shape[0])
        else:
            self.Psi_, factor = self._rebuild_with_qr_or_cholesky()
            if self.ortho_method == "qr":
                if self.R_rank_tol > 0.0:
                    self.RL_inv_T_, self.eff_rank_ = _safe_inverse_factor_qr(
                        factor, self.R_rank_tol,
                    )
                else:
                    self.RL_inv_T_ = jnp.linalg.inv(factor.T)
                    self.eff_rank_ = int(factor.shape[0])
            else:
                if self.R_rank_tol > 0.0:
                    self.RL_inv_T_, self.eff_rank_ = (
                        _safe_inverse_factor_cholesky(
                            factor, self.R_rank_tol,
                        )
                    )
                else:
                    self.RL_inv_T_ = jnp.linalg.inv(factor)
                    self.eff_rank_ = int(factor.shape[0])

        # --- 5. Global active set and final lifting matrix ----------------
        E_final = S_tilde - self.U_r_ @ self.S_hat_                    # (d, n)
        P_nl_final = self.Psi_[self.n_linear_:]                         # (m_nl, n)
        self.m_nl_ = int(P_nl_final.shape[0])

        C_nl = (1.0 / n) * (E_final @ P_nl_final.T)                     # (d, m_nl)
        self.importances_ = jnp.sum(C_nl ** 2, axis=0)                  # (m_nl,)

        threshold = self.gamma_sp / n
        norms = jnp.sqrt(self.importances_)
        active_nl_mask = norms > threshold
        active_nl_local = jnp.where(active_nl_mask)[0]
        global_nl_active = self.n_linear_ + active_nl_local
        self.active_nl_global_ = global_nl_active

        P_active = self.Psi_[global_nl_active]                          # (|A|, n)
        self.Vbar_nl_ = (1.0 / (n + self.gamma)) * (E_final @ P_active.T)

        linear_indices = jnp.arange(self.n_linear_)
        self.selected_ = jnp.concatenate([linear_indices, global_nl_active])
        self.Psi_selected_ = self.Psi_[self.selected_]

        # Cache normalization params (used only for the classical-family paths)
        if self.ortho_method != "arnoldi" and self.basis == "hybrid":
            self._store_normalization_params(self.S_hat_)

        self._fitted = True
        return self

    # =====================================================================
    # Rebuild helpers (QR / Cholesky paths)
    # =====================================================================

    def _evaluate_raw_features(self, S_hat: jnp.ndarray) -> jnp.ndarray:
        """Evaluate raw polynomial features for QR/Cholesky rebuild."""
        if self.basis == "hierarchical" or self.family == "monomial":
            return evaluate_monomials(S_hat, self.multi_indices_)
        max_deg = int(jnp.max(self.multi_indices_))
        return evaluate_classical_tensor(
            S_hat, self.multi_indices_, self.family, max_deg,
        )

    def _rebuild_with_qr_or_cholesky(
        self,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Rebuild Psi from scratch at the committed S_hat."""
        Phi = self._evaluate_raw_features(self.S_hat_)
        n_reorth = max(0, int(self.n_reorth))
        if self.ortho_method == "qr":
            if n_reorth > 0:
                return orthogonalize_qr_refined(Phi, n_reorth)
            return orthogonalize_qr(Phi)
        # cholesky
        if n_reorth > 0:
            return orthogonalize_cholesky_refined(Phi, n_reorth)
        return orthogonalize_cholesky(Phi)

    def _store_normalization_params(self, S_hat: jnp.ndarray) -> None:
        """Cache rescaling parameters used by classical tensor evaluation."""
        if self.family == "hermite":
            self._norm_params = {
                "type": "hermite",
                "mu": jnp.mean(S_hat, axis=1, keepdims=True),
                "sigma": jnp.maximum(
                    jnp.std(S_hat, axis=1, keepdims=True), 1e-15,
                ),
            }
        else:
            a = jnp.min(S_hat, axis=1, keepdims=True)
            b = jnp.max(S_hat, axis=1, keepdims=True)
            self._norm_params = {
                "type": self.family,
                "a": a,
                "b": b,
                "rng": jnp.maximum(b - a, 1e-15),
            }

    # =====================================================================
    # Feature evaluation at new points
    # =====================================================================

    def _evaluate_orthonormal_features(
        self, S_hat: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return Psi at new coordinates using the fitted method."""
        if self.ortho_method == "arnoldi":
            return arnoldi_eval(S_hat, self.arnoldi_basis_)
        Phi_new = self._evaluate_raw_features_for_decode(S_hat)
        return self.RL_inv_T_ @ Phi_new

    def _evaluate_raw_features_for_decode(
        self, S_hat: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.basis == "hierarchical" or self.family == "monomial":
            return evaluate_monomials(S_hat, self.multi_indices_)
        return self._evaluate_classical_with_stored_norm(S_hat)

    def _evaluate_classical_with_stored_norm(
        self, S_hat: jnp.ndarray,
    ) -> jnp.ndarray:
        max_deg = int(jnp.max(self.multi_indices_))
        if self._norm_params["type"] == "hermite":
            xi = (
                (S_hat - self._norm_params["mu"])
                / self._norm_params["sigma"]
            )
        else:
            xi = (
                2.0 * (S_hat - self._norm_params["a"])
                / self._norm_params["rng"]
                - 1.0
            )
            if self._norm_params["type"] == "chebyshev":
                xi = xi * 0.999

        eval_fn = functools.partial(
            _FAMILY_DISPATCH[self.family], max_deg=max_deg,
        )
        uni_vals = jax.vmap(eval_fn, in_axes=0, out_axes=0)(xi)
        r_ = S_hat.shape[0]
        dims = jnp.arange(r_)
        gathered = uni_vals[dims[None, :], self.multi_indices_]
        return jnp.prod(gathered, axis=1)

    # =====================================================================
    # encode / decode / reconstruct
    # =====================================================================

    def encode(self, S: jnp.ndarray) -> jnp.ndarray:
        """Project full-state snapshots onto the selected basis."""
        squeeze = S.ndim == 1
        if squeeze:
            S = S[:, None]
        s_hat = self.U_r_.T @ (S - self.s_ref_)
        return s_hat.squeeze(axis=1) if squeeze else s_hat

    def decode(self, S_hat: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct full-state from reduced coordinates."""
        squeeze = S_hat.ndim == 1
        if squeeze:
            S_hat = S_hat[:, None]

        Psi_new = self._evaluate_orthonormal_features(S_hat)
        Psi_nl_active = Psi_new[self.active_nl_global_]
        S_approx = (
            self.s_ref_
            + self.U_r_ @ S_hat
            + self.Vbar_nl_ @ Psi_nl_active
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
        S_approx = self.reconstruct(S)
        err = float(jnp.linalg.norm(S - S_approx, "fro"))
        if relative:
            return err / float(jnp.linalg.norm(S - self.s_ref_, "fro"))
        return err

    def linear_error(
        self, S: jnp.ndarray, *, relative: bool = True,
    ) -> float:
        S_centered = S - self.s_ref_
        S_hat = self.U_r_.T @ S_centered
        S_linear = self.s_ref_ + self.U_r_ @ S_hat
        err = float(jnp.linalg.norm(S - S_linear, "fro"))
        if relative:
            return err / float(jnp.linalg.norm(S_centered, "fro"))
        return err

    def energy_captured(self) -> dict:
        """Energy breakdown: POD + nonlinear + residual."""
        total = float(jnp.sum(self.sigma_s_ ** 2))
        pod = float(jnp.sum(self.sigma_s_[jnp.asarray(self.J_)] ** 2))
        n = self.Psi_.shape[1]
        threshold = self.gamma_sp / n
        active = jnp.sqrt(self.importances_) > threshold
        nl = float(n * jnp.sum(self.importances_ * active))
        residual = max(total - pod - nl, 0.0)
        return {
            "total": total, "pod": pod, "nonlinear": nl,
            "residual": residual,
            "pod_frac": pod / total if total > 0 else 1.0,
            "nl_frac": nl / total if total > 0 else 0.0,
        }

    @property
    def n_selected_nonlinear(self) -> int:
        return int(self.active_nl_global_.shape[0])

    @property
    def n_total_features(self) -> int:
        return int(self.selected_.shape[0])

    @property
    def selected_modes(self) -> np.ndarray:
        """Indices of the greedily selected singular vectors."""
        return self.J_

    def summary(self) -> dict:
        m_total = int(self.Psi_.shape[0])
        return {
            "r": self.r,
            "s": self.s,
            "p": self.p,
            "degrees": self.degrees_,
            "selected_modes": [int(j) for j in self.J_],
            "n_snapshots": int(self.Psi_.shape[1]),
            "state_dim": int(self.U_r_.shape[0]),
            "total_poly_features": m_total,
            "effective_rank": getattr(self, "eff_rank_", m_total),
            "linear_features": int(self.n_linear_),
            "selected_nonlinear": self.n_selected_nonlinear,
            "total_selected": self.n_total_features,
            "gamma_sp": self.gamma_sp,
            "gamma": self.gamma,
            "family": self.family,
            "basis": self.basis,
            "ortho_method": self.ortho_method,
        }

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        modes = list(self.J_) if self._fitted else "?"
        return (
            f"GreedyOPM(r={self.r}, s={self.s}, p={self.p}, "
            f"degrees={self.degrees_}, ortho={self.ortho_method!r}, "
            f"modes={modes}, {status})"
        )


# =============================================================================
# GreedyFixedOPM  --  Schwerdtner & Peherstorfer greedy with no sparsification
# =============================================================================


@jax.jit
def _ortho_error_from_features(
    sigma_out: jnp.ndarray,      # (s - i,)
    Wt_out_Pt: jnp.ndarray,      # (s - i, m_nl)
    gamma: float,
    n: int,
) -> jnp.ndarray:
    """Manifold reconstruction error  ||E - Vbar P||_F^2  in the reduced space."""
    linear_residual = jnp.sum(sigma_out ** 2)
    BP = sigma_out[:, None] * Wt_out_Pt                  # (s - i, m_nl)
    nl_correction = jnp.sum(BP ** 2) / (n + gamma)
    return linear_residual - nl_correction


class GreedyFixedOPM:
    """Greedy mode selection with an ALL-ACTIVE orthogonal polynomial feature map.

    Mirrors Schwerdtner & Peherstorfer (2024) but uses empirically orthonormal
    polynomial features.  No sparsification: all nonlinear features are used
    in the lifting matrix.

    Parameters
    ----------
    r, s, p : as in :class:`GreedyOPM`.
    gamma : float
        Tikhonov regularisation for the lifting matrix.
    n_reorth : int
        Number of additional Gram-Schmidt passes.
    R_rank_tol : float
        Rank truncation tolerance (QR / Cholesky paths only).
    family : as in :class:`GreedyOPM`.
    ortho_method : as in :class:`GreedyOPM`.  Used both in the candidate
        evaluation loop and the post-selection rebuild.
    center : bool
    """

    def __init__(
        self,
        r: int,
        s: int,
        p: Union[int, Sequence[int]] = 2,
        *,
        gamma: float = 1e-14,
        n_reorth: int = 1,
        R_rank_tol: float = 0.0,
        family: Literal["monomial", "legendre", "chebyshev", "hermite"] = "monomial",
        ortho_method: Literal["qr", "cholesky", "arnoldi"] = "arnoldi",
        center: bool = True,
    ):
        if ortho_method not in _ORTHO_METHODS:
            raise ValueError(
                f"ortho_method must be one of {_ORTHO_METHODS}; got {ortho_method!r}."
            )
        _validate_ortho_family(ortho_method, family, "GreedyFixedOPM")

        self.r = r
        self.s = s
        self.p = p
        self.degrees_ = _normalize_degrees(p)
        self.p_max_ = max(self.degrees_)
        self.gamma = gamma
        self.n_reorth = n_reorth
        self.R_rank_tol = R_rank_tol
        self.family = family
        self.ortho_method = ortho_method
        self.center = center
        self._fitted = False

    # ----- Degree filtering --------------------------------------------------

    @staticmethod
    def _effective_degrees(
        n_coords: int, n: int, degrees: Sequence[int],
    ) -> list:
        """Filter degrees so total feature count <= n snapshots."""
        from math import comb
        eff = list(degrees)
        while eff:
            total = 1 + n_coords + sum(
                comb(n_coords + q - 1, q) for q in eff
            )
            if total <= n:
                return eff
            eff = eff[:-1]
        return eff

    # ----- Feature construction for candidate evaluation ---------------------

    def _build_orthonormal_features(
        self,
        trial_Shat: jnp.ndarray,
        multi_indices_jnp: jnp.ndarray,
        multi_indices_np: np.ndarray,
        max_deg: int,
    ) -> jnp.ndarray:
        """Produce Psi on trial coordinates."""
        if self.ortho_method == "arnoldi":
            Psi, _ = arnoldi_build(
                trial_Shat, multi_indices_np,
                n_reorth=max(1, int(self.n_reorth)),
            )
            return Psi
        if self.family == "monomial":
            Phi = evaluate_monomials(trial_Shat, multi_indices_jnp)
        else:
            Phi = evaluate_classical_tensor(
                trial_Shat, multi_indices_jnp, self.family, max_deg,
            )
        if self.ortho_method == "qr":
            if self.n_reorth > 0:
                Psi, _ = orthogonalize_qr_refined(Phi, int(self.n_reorth))
            else:
                Psi, _ = orthogonalize_qr(Phi)
        else:
            if self.n_reorth > 0:
                Psi, _ = orthogonalize_cholesky_refined(Phi, int(self.n_reorth))
            else:
                Psi, _ = orthogonalize_cholesky(Phi)
        return Psi

    # ----- Single-candidate error -------------------------------------------

    def _candidate_error(
        self,
        trial_Shat: jnp.ndarray,
        sigma_out: jnp.ndarray,
        W_out: jnp.ndarray,
        multi_indices_jnp: jnp.ndarray,
        multi_indices_np: np.ndarray,
        max_deg: int,
        nl_start: int,
        n: int,
    ) -> float:
        Psi = self._build_orthonormal_features(
            trial_Shat, multi_indices_jnp, multi_indices_np, max_deg,
        )
        P_nl = Psi[nl_start:]
        Wt_out_Pt = W_out.T @ P_nl.T
        return float(_ortho_error_from_features(
            sigma_out, Wt_out_Pt, self.gamma, n,
        ))

    # ----- fit ---------------------------------------------------------------

    def fit(self, S: jnp.ndarray) -> "GreedyFixedOPM":
        d, n = S.shape

        if self.center:
            self.s_ref_ = jnp.mean(S, axis=1, keepdims=True)
        else:
            self.s_ref_ = jnp.zeros((d, 1), dtype=S.dtype)
        S_tilde = S - self.s_ref_

        U_full, sigma_full, Wt_full = jnp.linalg.svd(
            S_tilde, full_matrices=False,
        )
        s_eff = min(self.s, int(len(sigma_full)))
        U_s = U_full[:, :s_eff]
        sigma_s = sigma_full[:s_eff]
        W_s = Wt_full[:s_eff, :].T
        self.sigma_s_ = sigma_s
        SigmaWt = sigma_s[:, None] * W_s.T

        J: list[int] = []
        J_complement: set[int] = set(range(s_eff))

        n_f = jnp.asarray(float(n), dtype=S.dtype)
        gamma_f = jnp.asarray(float(self.gamma), dtype=S.dtype)
        n_reorth_eff = max(1, int(self.n_reorth))

        for i in range(self.r):
            n_coords = i + 1
            p_eff = self._effective_degrees(n_coords, n, self.degrees_)
            if not p_eff:
                all_mi_np = graded_lex_multi_indices_for_degrees(
                    n_coords, [],
                )
            else:
                all_mi_np = graded_lex_multi_indices_for_degrees(
                    n_coords, p_eff,
                )
            all_mi_np = np.asarray(all_mi_np)
            all_mi_jnp = jnp.array(all_mi_np)
            max_deg = int(np.max(all_mi_np)) if len(all_mi_np) > 0 else 0
            nl_start = 1 + n_coords

            candidates = sorted(J_complement)

            if self.ortho_method == "arnoldi":
                # --- Batched path: build all candidate Psi at once -------
                candidates_arr = jnp.asarray(candidates, dtype=jnp.int32)
                n_cand = int(candidates_arr.shape[0])

                cand_row = SigmaWt[candidates_arr][:, None, :]
                if i == 0:
                    trial_S_hat_all = cand_row
                else:
                    S_hat_old = SigmaWt[jnp.asarray(J)]
                    S_hat_old_b = jnp.broadcast_to(
                        S_hat_old[None, :, :], (n_cand, i, n),
                    )
                    trial_S_hat_all = jnp.concatenate(
                        [S_hat_old_b, cand_row], axis=1,
                    )

                Psi_all, _, _, _ = arnoldi_build_batch(
                    trial_S_hat_all, all_mi_np, n_reorth=n_reorth_eff,
                )                                                          # (n_cand, M, n)
                P_nl_all = Psi_all[:, nl_start:, :]                        # (n_cand, m_nl, n)

                # For each candidate c, trial = J ∪ {cand[c]},
                # out(c) = complement \ {cand[c]} = all other candidates.
                # Maximise:  sigma_{cand[c]}^2 + (1/(n+gamma)) * A(c),
                # where A(c) = sum_{c' != c, k} (SigmaWt[cands[c']] · P_nl[c, k])^2.
                w_P_nl_all = jnp.einsum(
                    "nl,cmn->clm", W_s, P_nl_all,
                )                                                          # (n_cand, s, m_nl)
                w_P_at_cands = w_P_nl_all[:, candidates_arr, :]            # (n_cand, n_cand, m_nl)
                sigma_cands = sigma_s[candidates_arr]
                scaled = sigma_cands[None, :, None] * w_P_at_cands          # (n_cand, n_cand, m_nl)
                sq = scaled ** 2
                total = jnp.sum(sq, axis=(1, 2))                           # (n_cand,)
                diag_ = jnp.sum(
                    jnp.diagonal(sq, axis1=0, axis2=1), axis=0,
                )                                                          # (n_cand,)
                A_c = total - diag_
                score = sigma_cands ** 2 + A_c / (n_f + gamma_f)
                best_local = int(jnp.argmax(score))
                best_j = candidates[best_local]
            else:
                # --- QR / Cholesky path: per-candidate Python loop --------
                best_j = candidates[0]
                best_error = jnp.inf
                for j_prime in candidates:
                    trial_idx = jnp.array(J + [j_prime], dtype=jnp.int32)
                    trial_Shat = SigmaWt[trial_idx]
                    Jc = sorted(J_complement - {j_prime})
                    Jc_arr = jnp.array(Jc, dtype=jnp.int32)

                    error = self._candidate_error(
                        trial_Shat,
                        sigma_s[Jc_arr],
                        W_s[:, Jc_arr],
                        all_mi_jnp, all_mi_np, max_deg, nl_start, n,
                    )
                    if error < float(best_error):
                        best_error = error
                        best_j = j_prime

            J.append(best_j)
            J_complement.remove(best_j)

        # ---- Post-selection rebuild ----------------------------------------
        self.J_ = np.asarray(J, dtype=np.int32)
        self.U_r_ = U_s[:, jnp.asarray(J)]
        self.S_hat_ = SigmaWt[jnp.asarray(J)]

        deg_eff = self._effective_degrees(self.r, n, self.degrees_)
        self.degrees_eff_ = deg_eff
        if deg_eff != self.degrees_:
            warnings.warn(
                f"Requested degrees {self.degrees_} exceed snapshot count. "
                f"Using effective degrees {deg_eff}.",
                stacklevel=2,
            )

        all_indices_np = graded_lex_multi_indices_for_degrees(
            self.r, deg_eff,
        )
        all_indices_np = np.asarray(all_indices_np)
        self.multi_indices_ = jnp.array(all_indices_np)
        self._multi_indices_np_ = all_indices_np

        max_deg_final = (
            int(np.max(all_indices_np)) if len(all_indices_np) > 0 else 0
        )

        self.arnoldi_basis_: Optional[ArnoldiBasis] = None
        self.RL_inv_T_: Optional[jnp.ndarray] = None

        if self.ortho_method == "arnoldi":
            self.Psi_, self.arnoldi_basis_ = arnoldi_build(
                self.S_hat_, all_indices_np,
                n_reorth=max(1, int(self.n_reorth)),
            )
            self.eff_rank_ = int(self.Psi_.shape[0])
        else:
            if self.family == "monomial":
                Phi = evaluate_monomials(self.S_hat_, self.multi_indices_)
            else:
                Phi = evaluate_classical_tensor(
                    self.S_hat_, self.multi_indices_,
                    self.family, max_deg_final,
                )
                self._store_normalization_params(self.S_hat_)

            if self.ortho_method == "qr":
                if self.n_reorth > 0:
                    self.Psi_, factor = orthogonalize_qr_refined(
                        Phi, int(self.n_reorth),
                    )
                else:
                    self.Psi_, factor = orthogonalize_qr(Phi)
                if self.R_rank_tol > 0.0:
                    self.RL_inv_T_, self.eff_rank_ = _safe_inverse_factor_qr(
                        factor, self.R_rank_tol,
                    )
                else:
                    self.RL_inv_T_ = jnp.linalg.inv(factor.T)
                    self.eff_rank_ = int(factor.shape[0])
            else:
                if self.n_reorth > 0:
                    self.Psi_, factor = orthogonalize_cholesky_refined(
                        Phi, int(self.n_reorth),
                    )
                else:
                    self.Psi_, factor = orthogonalize_cholesky(Phi)
                if self.R_rank_tol > 0.0:
                    self.RL_inv_T_, self.eff_rank_ = (
                        _safe_inverse_factor_cholesky(
                            factor, self.R_rank_tol,
                        )
                    )
                else:
                    self.RL_inv_T_ = jnp.linalg.inv(factor)
                    self.eff_rank_ = int(factor.shape[0])

        # ---- Final lifting matrix ------------------------------------------
        slices = degree_slices_for_degrees(self.r, deg_eff)
        self.n_linear_ = slices[1].stop if 1 in slices else slices[0].stop

        E_final = S_tilde - self.U_r_ @ self.S_hat_
        P_nl_final = self.Psi_[self.n_linear_:]
        self.m_nl_ = int(P_nl_final.shape[0])
        self.Vbar_nl_ = (1.0 / (n + self.gamma)) * (E_final @ P_nl_final.T)
        self.active_nl_global_ = jnp.arange(
            self.n_linear_, self.Psi_.shape[0],
        )

        self._fitted = True
        return self

    def _store_normalization_params(self, S_hat: jnp.ndarray) -> None:
        if self.family == "hermite":
            self._norm_params = {
                "type": "hermite",
                "mu": jnp.mean(S_hat, axis=1, keepdims=True),
                "sigma": jnp.maximum(
                    jnp.std(S_hat, axis=1, keepdims=True), 1e-15,
                ),
            }
        else:
            a = jnp.min(S_hat, axis=1, keepdims=True)
            b = jnp.max(S_hat, axis=1, keepdims=True)
            self._norm_params = {
                "type": self.family,
                "a": a, "b": b,
                "rng": jnp.maximum(b - a, 1e-15),
            }

    # ----- encode / decode / reconstruct ------------------------------------

    def encode(self, S: jnp.ndarray) -> jnp.ndarray:
        squeeze = S.ndim == 1
        if squeeze:
            S = S[:, None]
        s_hat = self.U_r_.T @ (S - self.s_ref_)
        return s_hat.squeeze(axis=1) if squeeze else s_hat

    def decode(self, S_hat: jnp.ndarray) -> jnp.ndarray:
        squeeze = S_hat.ndim == 1
        if squeeze:
            S_hat = S_hat[:, None]

        Psi_new = self._evaluate_orthonormal_features(S_hat)
        Psi_nl = Psi_new[self.active_nl_global_]
        S_approx = (
            self.s_ref_ + self.U_r_ @ S_hat + self.Vbar_nl_ @ Psi_nl
        )
        return S_approx.squeeze(axis=1) if squeeze else S_approx

    def reconstruct(self, S: jnp.ndarray) -> jnp.ndarray:
        return self.decode(self.encode(S))

    def _evaluate_orthonormal_features(
        self, S_hat: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.ortho_method == "arnoldi":
            return arnoldi_eval(S_hat, self.arnoldi_basis_)
        Phi_new = self._evaluate_raw_features_for_decode(S_hat)
        return self.RL_inv_T_ @ Phi_new

    def _evaluate_raw_features_for_decode(
        self, S_hat: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.family == "monomial":
            return evaluate_monomials(S_hat, self.multi_indices_)
        return self._evaluate_classical_with_stored_norm(S_hat)

    def _evaluate_classical_with_stored_norm(
        self, S_hat: jnp.ndarray,
    ) -> jnp.ndarray:
        max_deg = int(jnp.max(self.multi_indices_))
        if self._norm_params["type"] == "hermite":
            xi = (
                (S_hat - self._norm_params["mu"])
                / self._norm_params["sigma"]
            )
        else:
            xi = (
                2.0 * (S_hat - self._norm_params["a"])
                / self._norm_params["rng"]
                - 1.0
            )
            if self._norm_params["type"] == "chebyshev":
                xi = xi * 0.999

        eval_fn = functools.partial(
            _FAMILY_DISPATCH[self.family], max_deg=max_deg,
        )
        uni_vals = jax.vmap(eval_fn, in_axes=0, out_axes=0)(xi)
        r_ = S_hat.shape[0]
        dims = jnp.arange(r_)
        gathered = uni_vals[dims[None, :], self.multi_indices_]
        return jnp.prod(gathered, axis=1)

    # ----- Diagnostics ------------------------------------------------------

    def reconstruction_error(
        self, S: jnp.ndarray, *, relative: bool = True,
    ) -> float:
        S_approx = self.reconstruct(S)
        err = float(jnp.linalg.norm(S - S_approx, "fro"))
        if relative:
            return err / float(jnp.linalg.norm(S - self.s_ref_, "fro"))
        return err

    def linear_error(
        self, S: jnp.ndarray, *, relative: bool = True,
    ) -> float:
        S_centered = S - self.s_ref_
        S_hat = self.U_r_.T @ S_centered
        S_linear = self.s_ref_ + self.U_r_ @ S_hat
        err = float(jnp.linalg.norm(S - S_linear, "fro"))
        if relative:
            return err / float(jnp.linalg.norm(S_centered, "fro"))
        return err

    @property
    def selected_modes(self) -> np.ndarray:
        return self.J_

    def summary(self) -> dict:
        m_total = int(self.Psi_.shape[0])
        return {
            "r": self.r,
            "s": self.s,
            "p": self.p,
            "degrees": self.degrees_,
            "degrees_eff": getattr(self, "degrees_eff_", self.degrees_),
            "selected_modes": [int(j) for j in self.J_],
            "n_snapshots": int(self.Psi_.shape[1]),
            "state_dim": int(self.U_r_.shape[0]),
            "total_poly_features": m_total,
            "effective_rank": getattr(self, "eff_rank_", m_total),
            "linear_features": int(self.n_linear_),
            "nonlinear_features": int(self.m_nl_),
            "gamma": self.gamma,
            "family": self.family,
            "ortho_method": self.ortho_method,
        }

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        modes = list(self.J_) if self._fitted else "?"
        return (
            f"GreedyFixedOPM(r={self.r}, s={self.s}, p={self.p}, "
            f"degrees={self.degrees_}, "
            f"family={self.family!r}, ortho={self.ortho_method!r}, "
            f"modes={modes}, {status})"
        )