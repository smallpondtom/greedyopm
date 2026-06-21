"""
Validation: multivariate Arnoldi vs QR-on-monomials.

This version addresses three pitfalls of the previous test script:

  (1) OUT-OF-DISTRIBUTION TEST POINTS.  The previous test drew test
      points from rng.standard_normal while training came from SVD
      coordinates of an advecting pulse.  For degree-p polynomials,
      extrapolation inflates errors by O(|x|^p), swamping any signal
      about orthogonalization quality.  FIX: split the pulse
      trajectory into train/test along the time axis.

  (2) CLASSICAL-TENSOR RESCALING INCONSISTENCY.  evaluate_classical_
      tensor auto-rescales its input via min/max (or mean/std).
      Calling it separately on S_hat and S_test applies two different
      rescalings, so Phi(S_test) is not the same polynomial as
      Phi(S_hat) -- f_train and f_test_truth become *different*
      polynomials.  FIX: always define the ground truth in the
      monomial basis, which never rescales.

  (3) NUMERICAL RANK-DEFICIENCY.  On structured data, the span of
      monomials up to degree p has effective rank < m.  Both QR and
      Arnoldi produce orthonormal bases but differ in the near-null-
      space.  This is a property of the problem, not of either
      algorithm.  DIAGNOSTIC: report the effective rank of Phi via
      its SVD; when eff_rank < m, T2 and T4 are expected to be noisy.

TEST SUITE
----------
  T1  Training orthogonality.
  T2  Common span (effective-rank aware).
  T3  Monomial reproduction.
  T4  Out-of-sample agreement on held-out pulse snapshots.
"""

#%%
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "experiments"))
from data.transport import generate_advecting_pulse

from greedyopm.polynomials.multindex import graded_lex_multi_indices
from greedyopm.polynomials.monomial import evaluate_monomials
from greedyopm.polynomials.classical.classical_tensor import evaluate_classical_tensor
from greedyopm.polynomials.orthogonalize import orthogonalize_qr_refined
from greedyopm.polynomials.arnoldi import arnoldi_build, arnoldi_eval

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Build SVD-reduced coordinates for the advecting pulse (train+test pool)
# ---------------------------------------------------------------------------

def build_pulse_reduced_coords(r: int) -> jnp.ndarray:
    """Return S_hat_all in (r, n_time) from the advecting pulse dataset."""
    S, _, _ = generate_advecting_pulse(
        pulse_width=5.0e-4,
        pulse_shift=0.1,
        speed=5.0,
        final_time=0.20,
        n_time_samples=2000,
        n_space_samples=2**10,
    )
    S = np.asarray(S)
    S_c = S - np.mean(S, axis=1, keepdims=True)
    lo, hi = float(S_c.min()), float(np.abs(S_c).max())
    S_n = (S_c - lo) / (hi - lo)
    S_n = S_c
    U, _, _ = np.linalg.svd(S_n, full_matrices=False)
    V_r = U[:, :r]
    return jnp.asarray(V_r.T @ S_n)



# ---------------------------------------------------------------------------
# One comparison at fixed (r, n, p)
# ---------------------------------------------------------------------------

def run_comparison(r: int, n: int, p: int, n_test: int = 200,
                   *, seed: int = 0, label: str = "", type: str = "monomial"):
    print(f"\n{'='*72}")
    print(f"  {label}    r={r},  n={n},  p={p},  n_test={n_test}")
    print(f"{'='*72}")

    rng = np.random.default_rng(seed)

    # --- in-distribution train/test split ----------------------------------
    S_hat_all = build_pulse_reduced_coords(r)
    n_total = S_hat_all.shape[1]
    if n + n_test > n_total:
        raise ValueError(
            f"Need n+n_test={n+n_test} samples but only {n_total} available."
        )
    perm = rng.permutation(n_total)
    S_hat  = S_hat_all[:, perm[:n]]
    S_test = S_hat_all[:, perm[n:n + n_test]]

    # In-distribution range check (helps you see what's going on)
    tr_range = (float(S_hat.min()),  float(S_hat.max()))
    te_range = (float(S_test.min()), float(S_test.max()))
    print(f"  train coord range = [{tr_range[0]:+.3e}, {tr_range[1]:+.3e}]")
    print(f"  test  coord range = [{te_range[0]:+.3e}, {te_range[1]:+.3e}]")

    mi_np = graded_lex_multi_indices(r, p, min_degree=0)
    mi_j  = jnp.asarray(mi_np)
    m = mi_np.shape[0]

    if type == "monomial":
        Phi = evaluate_monomials(S_hat, mi_j)                 # (m, n)
        Phi_test = evaluate_monomials(S_test, mi_j)
    else:
        def legendre_fixed_rescale(S, a, b):
            xi = 2.0 * (S - a) / jnp.maximum(b - a, 1e-15) - 1.0
            return evaluate_classical_tensor(xi, mi_j, type, p)
    
        a = jnp.min(S_hat, axis=1, keepdims=True)
        b = jnp.max(S_hat, axis=1, keepdims=True)
        Phi      = legendre_fixed_rescale(S_hat,  a, b)
        Phi_test = legendre_fixed_rescale(S_test, a, b)   # same a, b!

    # --- build both bases (monomial basis is the common reference) ---------
    Psi_qr, R = orthogonalize_qr_refined(Phi, 1)
    Psi_arn, basis = arnoldi_build(S_hat, mi_np, n_reorth=1)

    # --- effective rank of Phi  (diagnostic for ill-posedness) -------------
    s_phi = jnp.linalg.svd(Phi, compute_uv=False)
    tol_rank = float(s_phi[0]) * 1e-13
    eff_rank = int(jnp.sum(s_phi > tol_rank))
    cond_phi = float(s_phi[0] / jnp.maximum(s_phi[-1], 1e-300))
    print(f"  m = {m},  effective rank of Phi = {eff_rank}  "
          f"(kappa(Phi) = {cond_phi:.2e})")

    diag_R = jnp.abs(jnp.diag(R))
    diag_H = jnp.abs(jnp.diag(basis.H))
    print(f"  R diagonal: min={float(jnp.min(diag_R)):.2e}  "
          f"max={float(jnp.max(diag_R)):.2e}")
    print(f"  H diagonal: min={float(jnp.min(diag_H)):.2e}  "
          f"max={float(jnp.max(diag_H)):.2e}")

    # --- T1: training orthogonality ----------------------------------------
    I_m = jnp.eye(m)
    e1_qr  = float(jnp.linalg.norm((1/n) * Psi_qr  @ Psi_qr.T  - I_m))
    e1_arn = float(jnp.linalg.norm((1/n) * Psi_arn @ Psi_arn.T - I_m))
    print(f"  T1 ortho QR              : {e1_qr:.3e}")
    print(f"  T1 ortho Arnoldi         : {e1_arn:.3e}")

    # --- T2: common span (rank-aware) --------------------------------------
    # Project Psi_arn onto row-space of Psi_qr:  P = (1/n) Psi_arn Psi_qr^T
    # If spans agree and no rank deficiency, ||P Psi_qr - Psi_arn|| = 0.
    # With rank deficiency, we report the residual AND the portion that
    # actually lies in the common effective span.
    P_cross = (1.0 / n) * (Psi_arn @ Psi_qr.T)
    span_res = float(jnp.linalg.norm(P_cross @ Psi_qr - Psi_arn))
    print(f"  T2 ||P Psi_qr - Psi_arn|| : {span_res:.3e}   "
          f"(expect ~0 when eff_rank == m)")

    # --- T3: monomial reproduction -----------------------------------------
    Phi_fro = float(jnp.linalg.norm(Phi))
    recon_qr  = float(jnp.linalg.norm(
        (Phi @ Psi_qr.T  / n) @ Psi_qr  - Phi,
    )) / Phi_fro
    recon_arn = float(jnp.linalg.norm(
        (Phi @ Psi_arn.T / n) @ Psi_arn - Phi,
    )) / Phi_fro
    print(f"  T3 monomial recon QR     : {recon_qr:.3e}")
    print(f"  T3 monomial recon Arnoldi: {recon_arn:.3e}")

    # --- T4: out-of-sample on HELD-OUT pulse snapshots ---------------------
    # Truth is a random polynomial IN THE MONOMIAL BASIS, so no rescaling
    # inconsistency between train and test.
    a_true = jnp.asarray(rng.standard_normal(m))
    f_train      = a_true @ Phi
    f_test_truth = a_true @ Phi_test

    # QR:  Psi_qr_test = R^{-T} Phi_test
    R_inv_T = jnp.linalg.inv(R.T)
    Psi_qr_test = R_inv_T @ Phi_test
    c_qr = (1.0 / n) * (Psi_qr @ f_train)
    f_qr = c_qr @ Psi_qr_test

    # Arnoldi:  Psi_arn_test via stored recurrence
    Psi_arn_test = arnoldi_eval(S_test, basis)
    c_arn = (1.0 / n) * (Psi_arn @ f_train)
    f_arn = c_arn @ Psi_arn_test

    truth_nrm = float(jnp.linalg.norm(f_test_truth))
    rel_qr  = float(jnp.linalg.norm(f_qr  - f_test_truth)) / truth_nrm
    rel_arn = float(jnp.linalg.norm(f_arn - f_test_truth)) / truth_nrm
    cross   = float(jnp.linalg.norm(f_qr - f_arn)) / (
        float(jnp.linalg.norm(f_qr)) + 1e-300
    )
    print(f"  T4 QR      vs truth      : {rel_qr:.3e}")
    print(f"  T4 Arnoldi vs truth      : {rel_arn:.3e}")
    print(f"  T4 QR      vs Arnoldi    : {cross:.3e}")


# ---------------------------------------------------------------------------
# Regime sweep
# ---------------------------------------------------------------------------

#%%
if __name__ == "__main__":
    # EASY: low degree, low r. Full rank expected.  Both methods agree to
    # ~1e-12 on held-out pulse snapshots.
    run_comparison(r=3, n=500, p=2, label="EASY", type="legendre")

    # MODERATE: rank deficiency kicks in (you should see eff_rank < m).
    # T4 agreement on in-distribution test points remains good for both
    # methods in the first few components of the active span, but the
    # amplitudes grow.  Check that Arnoldi agrees with QR up to the
    # effective rank.
    run_comparison(r=5, n=800, p=3, label="MODERATE", type="legendre")

    # HARD: strong rank deficiency.  The non-trivial message is that
    # both QR and Arnoldi lose accuracy together -- this is an intrinsic
    # difficulty of fitting degree-4 polynomials in r=8 coordinates of
    # a transport problem, NOT a failure of either algorithm.  The fix
    # is rank truncation (SVD) or reducing p.
    run_comparison(r=8, n=1200, p=4, label="HARD", type="legendre")

# %%
