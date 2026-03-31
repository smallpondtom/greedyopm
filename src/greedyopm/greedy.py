"""
Greedy Incremental Orthogonal Polynomial Manifold.

Implements the greedy algorithm that *jointly* selects
POD modes and polynomial features for nonlinear dimensionality reduction.

Key features:
  • Modes are selected greedily from a candidate pool of s > r singular
    vectors — they need not be the leading r.
  • At each greedy step, polynomial features are extended incrementally
    via an incremental QR factorization.
  • Feature activation uses group soft-thresholding, giving
    a one-step closed-form solution thanks to empirical orthonormality.
  • Old-feature contributions are updated via an O(1)-per-feature formula
    derived from the SVD structure.
  • All candidate evaluations operate in the (s - i)-dimensional SVD-reduced
    space, never forming d-dimensional matrices during the greedy loop.
"""

from __future__ import annotations

import functools
from typing import Dict, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from greedyopm.polynomials.multindex import (
    _new_nonlinear_multi_indices,
    graded_lex_multi_indices,
    degree_slices,
)
from greedyopm.polynomials.monomial import evaluate_monomials
from greedyopm.polynomials.classical_tensor import (
    evaluate_classical_tensor,
    _FAMILY_DISPATCH,
)
from greedyopm.polynomials.orthogonalize import (
    orthogonalize_qr,
    orthogonalize_cholesky,
)


# =============================================================================
# JIT-compiled core routines
# =============================================================================

@jax.jit
def _orthogonalize_new_linear(Q_old: jnp.ndarray,
                              new_coord: jnp.ndarray
                              ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Orthogonalise a new coordinate against Q_old columns.

    Parameters
    ----------
    Q_old : (n, q)  with Q_old^T Q_old = I_q
    new_coord : (n,)

    Returns
    -------
    q_new : (n,)  unit vector orthogonal to Q_old columns
    Q_new : (n, q+1)  updated orthonormal matrix
    """
    proj = Q_old @ (Q_old.T @ new_coord)
    perp = new_coord - proj
    norm = jnp.linalg.norm(perp)
    q_new = perp / jnp.maximum(norm, 1e-15)
    Q_new = jnp.concatenate([Q_old, q_new[:, None]], axis=1)
    return q_new, Q_new


@jax.jit
def _incremental_qr_new_features(
    Phi_new: jnp.ndarray,
    Q_old: jnp.ndarray,
    n: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Orthogonalise new monomial rows against existing orthonormal columns.

    Steps:
      1. Project out existing features:  Φ⊥ = Φ_new - Φ_new Q_old Q_old^T
      2. Thin QR:  (1/√n) (Φ⊥)^T = Q_new R_new
      3. P_new = √n Q_new^T   (new orthonormal feature rows)

    Parameters
    ----------
    Phi_new : (m_new, n)  raw monomial evaluations of new features
    Q_old   : (n, q)      existing orthonormal feature columns
    n       : int          number of snapshots

    Returns
    -------
    P_new   : (m_keep, n)  orthonormal new feature rows (may have m_keep < m_new
              if some features are linearly dependent)
    Q_new   : (n, m_keep)  new orthonormal columns to append to Q_old
    """
    # Step 1: project out existing features
    Phi_perp = Phi_new - (Phi_new @ Q_old) @ Q_old.T   # (m_new, n)

    # Step 2: thin QR
    F = Phi_perp.T / jnp.sqrt(n)                       # (n, m_new)
    Q_raw, R_raw = jnp.linalg.qr(F, mode="reduced")    # Q: (n, m_new), 
                                                       # R: (m_new, m_new)

    # Fix sign convention
    signs = jnp.sign(jnp.diag(R_raw))
    signs = jnp.where(signs == 0, 1.0, signs)
    Q_raw = Q_raw * signs[None, :]
    R_raw = signs[:, None] * R_raw

    # Step 3: orthonormal feature rows
    P_new = jnp.sqrt(n) * Q_raw.T                      # (m_new, n)
    return P_new, Q_raw


@jax.jit
def _compute_old_norms_sq(
    sigma_J: jnp.ndarray,     # (|J̆|,)
    w_P_J: jnp.ndarray,       # (|J̆|, m_nl)
    n: int,
) -> jnp.ndarray:
    """‖r̃_{i-1,k}‖² for all old features k.

    r̃_{i-1,k} = (1/n) Σ_{J̆} W_{J̆}^T [P]_{k,:}^T
    ‖r̃_{i-1,k}‖² = (1/n²) Σ_{ℓ∈J̆} σ_ℓ² · (wℓ^T p_k)²
    """
    return (1.0 / n**2) * jnp.sum(sigma_J[:, None]**2 * w_P_J**2, axis=0)


@jax.jit
def _compute_new_norms_sq(
    sigma_Jtrial: jnp.ndarray,   # (|J̆_trial|,)
    W_Jtrial: jnp.ndarray,       # (n, |J̆_trial|)
    P_new: jnp.ndarray,          # (m_new, n)
    n: int,
) -> jnp.ndarray:
    """‖r̃_k‖² for new features after adding mode j'.

    r̃_k = (1/n) diag(σ_{J̆}) W_{J̆}^T [P_new]_{k,:}^T
    """
    # (|J̆|, m_new)
    r_tilde = (1.0 / n) * (sigma_Jtrial[:, None] * (W_Jtrial.T @ P_new.T))
    return jnp.sum(r_tilde**2, axis=0)  # (m_new,)


# ============================================================================
# Main class
# ============================================================================

class GreedyOPM:
    """Greedy Incremental Orthogonal Polynomial Manifold.

    Parameters
    ----------
    r : int
        Target reduced dimension (number of modes to select).
    s : int
        Candidate pool size (s > r).  The first s left-singular vectors
        of the centred snapshot matrix form the candidate set.
    p : int
        Maximum total polynomial degree.
    gamma_sp : float
        Sparsity parameter for group soft-thresholding (γ_sp in the paper).
        Features with ‖r_k‖₂ ≤ γ_sp / n are deactivated.
        Set to 0.0 to keep all features active (no sparsity).
    gamma : float
        Tikhonov regulariation for the final lifting matrix
        (γ in equation 20 of the paper).
    family : str
        Classical polynomial family for hybrid basis construction.
    basis : str
        ``"hybrid"`` (classical + empirical QR) or ``"hierarchical"``
        (monomial + empirical QR).
    ortho_method : str
        ``"qr"`` or ``"cholesky"`` for the final feature rebuild.
    center : bool
        Whether to subtract the snapshot mean.
    """

    def __init__(
        self,
        r: int,
        s: int,
        p: int = 2,
        *,
        gamma_sp: float = 0.0,
        gamma: float = 1e-8,
        family: Literal["legendre", "chebyshev", "hermite"] = "legendre",
        basis: Literal["hybrid", "hierarchical"] = "hybrid",
        ortho_method: Literal["qr", "cholesky"] = "qr",
        center: bool = True,
    ):
        self.r = r
        self.s = s
        self.p = p
        self.gamma_sp = gamma_sp
        self.gamma = gamma
        self.family = family
        self.basis = basis
        self.ortho_method = ortho_method
        self.center = center
        self._fitted = False

    # ========================================================================
    # fit
    # ========================================================================

    def fit(self, S: jnp.ndarray) -> "GreedyOPM":
        """Fit the greedy OPM to snapshot data.

        Implements main algorithm: greedy mode selection with incremental
        polynomial feature extension and group soft-thresholding.

        Parameters
        ----------
        S : (d, n)  snapshot matrix  (d = state dim, n = # snapshots).

        Returns
        -------
        self
        """
        d, n = S.shape

        # ---- Step 1: centre -------------------------------------------------
        if self.center:
            self.s_ref_ = jnp.mean(S, axis=1, keepdims=True)   # (d, 1)
        else:
            self.s_ref_ = jnp.zeros((d, 1))
        S_tilde = S - self.s_ref_                              # (d, n)

        # ---- Step 2: rank-s truncated SVD -----------------------------------
        U_full, sigma_full, Wt_full = jnp.linalg.svd(
            S_tilde, full_matrices=False,
        )
        s = min(self.s, len(sigma_full))
        U_s = U_full[:, :s]            # (d, s)
        sigma_s = sigma_full[:s]       # (s,)
        W_s = Wt_full[:s, :].T         # (n, s)

        # Cache full SVD components needed later
        self.sigma_s_ = sigma_s

        # Precompute Σ Wᵀ: row j is σ_j · w_j^T
        SigmaWt = sigma_s[:, None] * W_s.T   # (s, n)

        # ---- Step 3: greedy mode selection loop -----------------------------
        J = []  # selected mode indices (ordered)
        J_set = set()
        J_complement = set(range(s))

        # Initialise Q_old with the degree-0 (constant) feature.
        # Q_old has standard-orthonormal columns: Q_old^T Q_old = I.
        Q_old = jnp.ones((n, 1)) / jnp.sqrt(n)

        # Current nonlinear feature rows P_nl: (m_nl, n)
        P_nl = jnp.empty((0, n))
        m_nl = 0

        # Precomputed inner products w_P[j, k] = w_j^T [P_nl]_{k,:}^T
        # for ALL s singular vectors (updated when P_nl grows).
        w_P = jnp.empty((s, 0))  # (s, m_nl)

        # Current ‖r̃_{i-1,k}‖² for each old nonlinear feature
        old_norms_sq = jnp.empty(0)  # (m_nl,)

        # Current active polynomial gain  Σ_{k∈A} n·‖r̃_k‖²
        current_poly_gain = 0.0

        for i in range(self.r):
            candidates = sorted(J_complement)
            n_cand = len(candidates)

            # -- Recompute old_norms_sq for the current J̆ ---------------------
            if m_nl > 0:
                J_compl_arr = jnp.array(candidates, dtype=jnp.int32)
                old_norms_sq = _compute_old_norms_sq(
                    sigma_s[J_compl_arr],
                    w_P[J_compl_arr],
                    n,
                )   # (m_nl,)

            best_j = candidates[0]
            best_score = -jnp.inf
            best_state = None

            for j_prime in candidates:
                # --- (a) linear gain: σ²_{j'} --------------------------------
                linear_gain = float(sigma_s[j_prime] ** 2)

                # --- (b) old features: rank-1 norm update --------------------
                # ‖r̃^{(j')}_{i,k}‖² = ‖r̃_{i-1,k}‖² − (σ_{j'}/n · w_{j'}^T p_k)²
                if m_nl > 0:
                    delta_old = (sigma_s[j_prime] / n)**2 * w_P[j_prime]**2
                    old_norms_sq_trial = jnp.maximum(old_norms_sq - delta_old, 0.0)
                else:
                    old_norms_sq_trial = jnp.empty(0)

                # --- (c) extend features incrementally -----------------------
                # (c1) orthogonalise new linear coordinate against Q_old
                new_coord = SigmaWt[j_prime]  # (n,)
                _, Q_trial_lin = _orthogonalize_new_linear(Q_old, new_coord)

                # (c2) generate new nonlinear multi-indices
                n_coords_trial = i + 1
                new_nl_np = _new_nonlinear_multi_indices(n_coords_trial, self.p)
                m_new = len(new_nl_np)

                if m_new > 0:
                    # build trial reduced coordinates
                    trial_idx = jnp.array(J + [j_prime], dtype=jnp.int32)
                    trial_Shat = SigmaWt[trial_idx]  # (n_coords_trial, n)

                    # evaluate new monomials
                    new_nl_jax = jnp.array(new_nl_np)
                    Phi_new = evaluate_monomials(trial_Shat, new_nl_jax)  # (m_new, n)

                    # orthogonalise against Q_trial_lin (includes all old + new linear)
                    P_new, Q_new_cols = _incremental_qr_new_features(
                        Phi_new, Q_trial_lin, n,
                    )
                    m_keep = P_new.shape[0]

                    # --- (d) new feature contributions -----------------------
                    J_trial_compl = sorted(J_complement - {j_prime})
                    J_tc_arr = jnp.array(J_trial_compl, dtype=jnp.int32)
                    new_norms_sq = _compute_new_norms_sq(
                        sigma_s[J_tc_arr], W_s[:, J_tc_arr], P_new, n,
                    )  # (m_keep,)
                else:
                    P_new = jnp.empty((0, n))
                    Q_new_cols = jnp.empty((n, 0))
                    new_norms_sq = jnp.empty(0)
                    m_keep = 0

                # --- (e) soft-threshold all features -------------------------
                all_norms_sq = jnp.concatenate([old_norms_sq_trial, new_norms_sq])
                threshold = self.gamma_sp / n
                all_norms = jnp.sqrt(jnp.maximum(all_norms_sq, 0.0))
                active_mask = all_norms > threshold

                # --- (f) polynomial gain with trial basis --------------------
                poly_gain_trial = float(n * jnp.sum(all_norms_sq * active_mask))

                # --- (g) ΔJ_mode ---------------------------------------------
                delta = linear_gain + poly_gain_trial - current_poly_gain

                if delta > float(best_score):
                    best_score = delta
                    best_j = j_prime
                    best_state = {
                        "P_new": P_new,
                        "Q_new_cols": Q_new_cols,
                        "active_mask": active_mask,
                        "poly_gain": poly_gain_trial,
                        "old_norms_sq_trial": old_norms_sq_trial,
                        "new_norms_sq": new_norms_sq,
                    }

            # ---- Commit best mode -------------------------------------------
            J.append(best_j)
            J_set.add(best_j)
            J_complement.remove(best_j)

            st = best_state

            # Update Q_old: append new linear column + new nonlinear columns
            _, Q_old = _orthogonalize_new_linear(Q_old, SigmaWt[best_j])
            if st["Q_new_cols"].shape[1] > 0:
                Q_old = jnp.concatenate([Q_old, st["Q_new_cols"]], axis=1)

            # Update P_nl
            if st["P_new"].shape[0] > 0:
                P_nl = (jnp.concatenate([P_nl, st["P_new"]], axis=0)
                        if m_nl > 0 else st["P_new"])
            m_nl = P_nl.shape[0]

            # Update w_P  (recompute — could be incremental for the new block)
            if m_nl > 0:
                w_P = W_s.T @ P_nl.T  # (s, m_nl)
            else:
                w_P = jnp.empty((s, 0))

            # Update current polynomial gain
            current_poly_gain = st["poly_gain"]

        # ---- Step 4: post-selection feature rebuild -------------------------
        # After greedy mode selection, rebuild the full polynomial features
        # for the final r-dimensional coordinates using the standard
        # orthogonalisation pipeline.  This gives a clean RL_inv_T for
        # out-of-sample evaluation.
        self.J_ = jnp.array(J, dtype=jnp.int32)
        self.U_r_ = U_s[:, J]                # (d, r)

        # Final reduced coordinates
        self.S_hat_ = self.U_r_.T @ S_tilde  # (r, n)

        # Build all multi-indices for the final r coordinates
        all_indices_np = graded_lex_multi_indices(self.r, self.p, min_degree=0)
        self.multi_indices_ = jnp.array(all_indices_np)

        # Evaluate raw features
        if self.basis == "hybrid":
            max_deg = int(np.max(all_indices_np))
            Phi_final = evaluate_classical_tensor(
                self.S_hat_, self.multi_indices_, self.family, max_deg,
            )
            self._store_normalization_params(self.S_hat_)
        else:
            Phi_final = evaluate_monomials(self.S_hat_, self.multi_indices_)

        # Empirical orthonormalisation
        if self.ortho_method == "qr":
            self.Psi_, R = orthogonalize_qr(Phi_final)
            self.RL_inv_T_ = jnp.linalg.inv(R.T)
        else:
            self.Psi_, L = orthogonalize_cholesky(Phi_final)
            self.RL_inv_T_ = jnp.linalg.inv(L)

        # ---- Step 5: feature selection on rebuilt features ------------------
        slices = degree_slices(self.r, self.p)
        self.n_linear_ = slices[1].stop if 1 in slices else slices[0].stop

        E_final = S_tilde - self.U_r_ @ self.S_hat_          # (d, n)
        P_nl_final = self.Psi_[self.n_linear_:]              # (m_nl_final, n)
        m_nl_final = P_nl_final.shape[0]

        # Feature importances: ‖r_k‖² = ‖(1/n) E [P]_{k,:}^T‖²
        C_nl = (1.0 / n) * (E_final @ P_nl_final.T)          # (d, m_nl_final)
        self.importances_ = jnp.sum(C_nl**2, axis=0)         # (m_nl_final,)

        # Group soft-thresholding
        threshold = self.gamma_sp / n
        norms = jnp.sqrt(self.importances_)
        active_nl_mask = norms > threshold

        active_nl_indices = jnp.where(active_nl_mask)[0]

        # Assemble final selected indices: [all linear | active nonlinear]
        linear_indices = jnp.arange(self.n_linear_)
        global_nl_active = self.n_linear_ + active_nl_indices
        self.selected_ = jnp.concatenate([linear_indices, global_nl_active])

        # ---- Step 6: final lifting matrix -----------------------------------
        # For nonlinear features:  V̄ = 1/(n + γ) · E · P_A^T
        P_active = self.Psi_[global_nl_active]                # (m_active, n)
        self.Vbar_nl_ = (1.0 / (n + self.gamma)) * (E_final @ P_active.T)

        # Pre-slice for fast reconstruction
        self.Psi_selected_ = self.Psi_[self.selected_]        # (m_sel, n)
        self.active_nl_global_ = global_nl_active

        self._fitted = True
        return self

    # =========================================================================
    # Normalisation helpers
    # =========================================================================

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

    # =========================================================================
    # encode / decode / reconstruct
    # =========================================================================

    def encode(self, S: jnp.ndarray) -> jnp.ndarray:
        """Project full-state snapshots onto the greedily selected basis.

        Parameters
        ----------
        S : (d, n) or (d,)

        Returns
        -------
        S_hat : (r, n) or (r,)
        """
        squeeze = S.ndim == 1
        if squeeze:
            S = S[:, None]
        s_hat = self.U_r_.T @ (S - self.s_ref_)
        return s_hat.squeeze(axis=1) if squeeze else s_hat

    def decode(self, S_hat: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct full-state from reduced coordinates via the manifold.

        s_approx = s_ref + U_r ŝ + V̄_nl · p_active(ŝ)

        Steps:
          1. Evaluate raw polynomial features Φ(ŝ).
          2. Apply stored orthonormalisation:  Ψ = RL_inv_T · Φ.
          3. Select active nonlinear features.
          4. Reconstruct:  s_approx = s_ref + U_r ŝ + V̄_nl · ψ_active.

        Parameters
        ----------
        S_hat : (r, n) or (r,)

        Returns
        -------
        S_approx : (d, n) or (d,)
        """
        squeeze = S_hat.ndim == 1
        if squeeze:
            S_hat = S_hat[:, None]

        # Raw features at new coordinates
        Phi_new = self._evaluate_raw_features(S_hat)         # (m_total, n_new)

        # Apply stored orthonormalisation
        Psi_new = self.RL_inv_T_ @ Phi_new                   # (m_total, n_new)

        # Select active nonlinear features
        Psi_nl_active = Psi_new[self.active_nl_global_]      # (m_active, n_new)

        # Reconstruct: linear part + nonlinear correction
        S_approx = self.s_ref_ + self.U_r_ @ S_hat + self.Vbar_nl_ @ Psi_nl_active

        return S_approx.squeeze(axis=1) if squeeze else S_approx

    def reconstruct(self, S: jnp.ndarray) -> jnp.ndarray:
        """Full encode → decode pipeline."""
        return self.decode(self.encode(S))

    # =========================================================================
    # Raw feature evaluation
    # =========================================================================

    def _evaluate_raw_features(self, S_hat: jnp.ndarray) -> jnp.ndarray:
        """Evaluate raw (pre-orthonormalisation) polynomial features."""
        if self.basis == "hierarchical":
            return evaluate_monomials(S_hat, self.multi_indices_)
        return self._evaluate_classical_with_stored_norm(S_hat)

    def _evaluate_classical_with_stored_norm(
        self, S_hat: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate classical tensor polynomials using training-time rescaling."""
        max_deg = int(jnp.max(self.multi_indices_))

        if self._norm_params["type"] == "hermite":
            xi = (S_hat - self._norm_params["mu"]) / self._norm_params["sigma"]
        else:
            xi = (
                2.0 * (S_hat - self._norm_params["a"]) / self._norm_params["rng"]
                - 1.0
            )
            if self._norm_params["type"] == "chebyshev":
                xi = xi * 0.999

        eval_fn = functools.partial(
            _FAMILY_DISPATCH[self.family], max_deg=max_deg,
        )
        uni_vals = jax.vmap(eval_fn, in_axes=0, out_axes=0)(xi)

        r = S_hat.shape[0]
        dims = jnp.arange(r)
        gathered = uni_vals[dims[None, :], self.multi_indices_]
        return jnp.prod(gathered, axis=1)

    # =================================================================
    # Error diagnostics
    # =================================================================

    def reconstruction_error(
        self, S: jnp.ndarray, *, relative: bool = True,
    ) -> float:
        """Frobenius reconstruction error on the polynomial manifold."""
        S_approx = self.reconstruct(S)
        err = float(jnp.linalg.norm(S - S_approx, "fro"))
        if relative:
            S_centered = S - self.s_ref_
            return err / float(jnp.linalg.norm(S_centered, "fro"))
        return err

    def linear_error(
        self, S: jnp.ndarray, *, relative: bool = True,
    ) -> float:
        """Frobenius error of the linear (POD) approximation alone."""
        S_centered = S - self.s_ref_
        S_hat = self.U_r_.T @ S_centered
        S_linear = self.s_ref_ + self.U_r_ @ S_hat
        err = float(jnp.linalg.norm(S - S_linear, "fro"))
        if relative:
            return err / float(jnp.linalg.norm(S_centered, "fro"))
        return err

    def energy_captured(self) -> dict:
        """Energy breakdown: POD vs. nonlinear correction vs. residual."""
        n = self.Psi_.shape[1]
        S_tilde = self.s_ref_ * 0  # placeholder — need original data
        # Instead, use singular values:
        total_energy = float(jnp.sum(self.sigma_s_**2))
        pod_energy = float(jnp.sum(self.sigma_s_[np.array(self.J_)]**2))

        # Nonlinear energy from importances
        nl_energy = float(n * jnp.sum(self.importances_[
            jnp.where(jnp.sqrt(self.importances_) > self.gamma_sp / n)[0]
        ]))

        residual = total_energy - pod_energy - nl_energy
        return {
            "total": total_energy,
            "pod": pod_energy,
            "nonlinear": nl_energy,
            "residual": max(residual, 0.0),
            "pod_frac": pod_energy / total_energy if total_energy > 0 else 1.0,
            "nl_frac": nl_energy / total_energy if total_energy > 0 else 0.0,
        }

    # =========================================================================
    # Properties / summary
    # =========================================================================

    @property
    def n_selected_nonlinear(self) -> int:
        return len(self.selected_) - self.n_linear_

    @property
    def n_total_features(self) -> int:
        return len(self.selected_)

    @property
    def selected_modes(self) -> jnp.ndarray:
        """Indices of the greedily selected singular vectors."""
        return self.J_

    def summary(self) -> dict:
        m_total = self.Psi_.shape[0]
        return {
            "r": self.r,
            "s": self.s,
            "p": self.p,
            "selected_modes": [int(j) for j in self.J_],
            "n_snapshots": self.Psi_.shape[1],
            "state_dim": self.U_r_.shape[0],
            "total_poly_features": m_total,
            "linear_features": self.n_linear_,
            "selected_nonlinear": self.n_selected_nonlinear,
            "total_selected": self.n_total_features,
            "gamma_sp": self.gamma_sp,
            "gamma": self.gamma,
            "family": self.family,
            "basis": self.basis,
        }

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        modes = list(self.J_) if self._fitted else "?"
        return (
            f"GreedyOPM(r={self.r}, s={self.s}, p={self.p}, "
            f"modes={modes}, {status})"
        )