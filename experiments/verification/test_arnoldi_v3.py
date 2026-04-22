"""
Validation: multivariate Arnoldi vs QR.

Revisions from the previous script:

  (A) DATA NORMALIZATION RESTORED.  The previous script had
      `S_n = S_c` overriding the [0, 1] rescaling, leaving S_hat in
      a range of ~100.  With that scale, the Arnoldi recurrence at
      out-of-training points multiplies polynomials of magnitude
      ~1 by coordinate values of ~50 at every level of the
      recursion, inflating intermediate roundoff.  FIX: keep the
      original normalization; optionally standardize S_hat further
      using training statistics.

  (B) TRUTH IS ALWAYS IN THE MONOMIAL BASIS.  Defining truth as
      `a_true @ Phi_legendre` gives the QR pipeline a built-in
      analytical cancellation (R^T R^{-T} = I) that is independent
      of numerical conditioning, which makes QR look perfect at
      machine precision while Arnoldi (which has no equivalent
      cancellation) shows any finite-precision deviation.  That's
      a test artifact, not an algorithm difference.  FIX: always
      define truth as a random combination of MONOMIALS.  QR and
      Arnoldi then face the same numerical task.

  (C) NEW DIAGNOSTIC: arnoldi_eval self-consistency at training.
      Re-evaluating Psi_arn at the training points via
      arnoldi_eval must return the same matrix that arnoldi_build
      produced.  Any discrepancy here would indicate a bug in the
      eval function (it does not, in practice — this catches a
      class of future regressions).

TEST SUITE
----------
  T0  arnoldi_eval self-consistency at training.
  T1  Training orthogonality.
  T2  Common span (residual of Psi_arn vs its projection onto Psi_qr).
  T3  Monomial reproduction.
  T4  Out-of-sample recovery of a monomial-basis polynomial.
"""

# %%
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
from greedyopm.polynomials.classical_tensor import evaluate_classical_tensor
from greedyopm.polynomials.orthogonalize import orthogonalize_qr_refined
from greedyopm.polynomials.arnoldi import arnoldi_build, arnoldi_eval

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Reduced coordinates from the pulse, with PROPER normalization
# ---------------------------------------------------------------------------

def build_pulse_reduced_coords(r: int, *, standardize: bool = True
                               ) -> jnp.ndarray:
    """Return S_hat_all in (r, n_time) with stable magnitudes.

    Parameters
    ----------
    r : reduced dimension
    standardize : if True, additionally rescale each reduced coordinate
        to zero mean / unit empirical std (helps Arnoldi conditioning
        for higher p).  Does not change the polynomial span.
    """
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
    lo = float(S_c.min())
    hi = float(np.abs(S_c).max())
    S_n = (S_c - lo) / (hi - lo)                       # [0, 1] scaling

    U, _, _ = np.linalg.svd(S_n, full_matrices=False)
    V_r = U[:, :r]
    S_hat = V_r.T @ S_n                                # (r, n_time)

    if standardize:
        mu = S_hat.mean(axis=1, keepdims=True)
        sd = np.maximum(S_hat.std(axis=1, keepdims=True), 1e-15)
        S_hat = (S_hat - mu) / sd                      # unit variance

    return jnp.asarray(S_hat)


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
    S_hat_all = build_pulse_reduced_coords(r, standardize=False)
    n_total = S_hat_all.shape[1]
    if n + n_test > n_total:
        raise ValueError(
            f"Need n+n_test={n+n_test} samples, only {n_total} available."
        )
    perm = rng.permutation(n_total)
    S_hat  = S_hat_all[:, perm[:n]]
    S_test = S_hat_all[:, perm[n:n + n_test]]

    tr = (float(S_hat.min()),  float(S_hat.max()))
    te = (float(S_test.min()), float(S_test.max()))
    print(f"  train coord range = [{tr[0]:+.3e}, {tr[1]:+.3e}]")
    print(f"  test  coord range = [{te[0]:+.3e}, {te[1]:+.3e}]")

    # --- multi-indices and monomial Phi ------------------------------------
    mi_np = graded_lex_multi_indices(r, p, min_degree=0)
    mi_j  = jnp.asarray(mi_np)
    m = mi_np.shape[0]

    if type == "monomial":
        Phi      = evaluate_monomials(S_hat,  mi_j)        # monomials only
        Phi_test = evaluate_monomials(S_test, mi_j)
    else:
        Phi      = evaluate_classical_tensor(S_hat,  mi_j, type, p)
        Phi_test = evaluate_classical_tensor(S_test, mi_j, type, p)

    # --- build both bases --------------------------------------------------
    Psi_qr,  R     = orthogonalize_qr_refined(Phi, 1)
    Psi_arn, basis = arnoldi_build(S_hat, mi_np, n_reorth=1)

    # Effective rank + conditioning diagnostics
    s_phi = jnp.linalg.svd(Phi, compute_uv=False)
    tol_rank = float(s_phi[0]) * 1e-13
    eff_rank = int(jnp.sum(s_phi > tol_rank))
    print(f"  m = {m},  eff. rank of Phi = {eff_rank},  "
          f"kappa(Phi) = {float(s_phi[0]/s_phi[-1]):.2e}")
    print(f"  R diag: min={float(jnp.min(jnp.abs(jnp.diag(R)))):.2e}  "
          f"max={float(jnp.max(jnp.abs(jnp.diag(R)))):.2e}")
    print(f"  H diag: min={float(jnp.min(jnp.abs(jnp.diag(basis.H)))):.2e}  "
          f"max={float(jnp.max(jnp.abs(jnp.diag(basis.H)))):.2e}")

    # --- T0: arnoldi_eval self-consistency at training --------------------
    Psi_arn_train_rebuilt = arnoldi_eval(S_hat, basis)
    t0 = float(jnp.linalg.norm(Psi_arn_train_rebuilt - Psi_arn))
    t0_rel = t0 / float(jnp.linalg.norm(Psi_arn))
    print(f"  T0 arnoldi_eval(train)   : {t0:.3e}  (rel: {t0_rel:.3e})")

    # --- T1: training orthogonality ---------------------------------------
    I_m = jnp.eye(m)
    e1_qr  = float(jnp.linalg.norm((1/n) * Psi_qr  @ Psi_qr.T  - I_m))
    e1_arn = float(jnp.linalg.norm((1/n) * Psi_arn @ Psi_arn.T - I_m))
    print(f"  T1 ortho QR              : {e1_qr:.3e}")
    print(f"  T1 ortho Arnoldi         : {e1_arn:.3e}")

    # --- T2: common span --------------------------------------------------
    P_cross = (1.0 / n) * (Psi_arn @ Psi_qr.T)
    t2 = float(jnp.linalg.norm(P_cross @ Psi_qr - Psi_arn))
    print(f"  T2 ||P Psi_qr - Psi_arn||: {t2:.3e}  "
          f"(~0 expected when eff_rank == m)")

    # --- T3: monomial reproduction ----------------------------------------
    Phi_fro = float(jnp.linalg.norm(Phi))
    r3_qr  = float(jnp.linalg.norm(
        (Phi @ Psi_qr.T  / n) @ Psi_qr  - Phi)) / Phi_fro
    r3_arn = float(jnp.linalg.norm(
        (Phi @ Psi_arn.T / n) @ Psi_arn - Phi)) / Phi_fro
    print(f"  T3 monomial recon QR     : {r3_qr:.3e}")
    print(f"  T3 monomial recon Arnoldi: {r3_arn:.3e}")

    # --- T4: out-of-sample on held-out pulse snapshots --------------------
    #   Truth is a random monomial-basis polynomial.  Both methods face
    #   the same task:  compute its coefficients in their orthonormal
    #   basis, then evaluate at held-out points.  No analytical shortcut.
    a_true = jnp.asarray(rng.standard_normal(m))
    f_train      = a_true @ Phi
    f_test_truth = a_true @ Phi_test

    R_inv_T     = jnp.linalg.inv(R.T)
    Psi_qr_test = R_inv_T @ Phi_test
    c_qr = (1.0 / n) * (Psi_qr @ f_train)
    f_qr = c_qr @ Psi_qr_test

    Psi_arn_test = arnoldi_eval(S_test, basis)
    c_arn = (1.0 / n) * (Psi_arn @ f_train)
    f_arn = c_arn @ Psi_arn_test

    truth_nrm = float(jnp.linalg.norm(f_test_truth))
    rel_qr  = float(jnp.linalg.norm(f_qr  - f_test_truth)) / truth_nrm
    rel_arn = float(jnp.linalg.norm(f_arn - f_test_truth)) / truth_nrm
    cross   = float(jnp.linalg.norm(f_qr - f_arn)) / (
        float(jnp.linalg.norm(f_qr)) + 1e-300)
    print(f"  T4 QR      vs truth      : {rel_qr:.3e}")
    print(f"  T4 Arnoldi vs truth      : {rel_arn:.3e}")
    print(f"  T4 QR      vs Arnoldi    : {cross:.3e}")


# %% -----------------------------------------------------------------------
if __name__ == "__main__":
    run_comparison(r=3, n=500,  p=2, label="EASY", type="monomial")
    run_comparison(r=5, n=800,  p=3, label="MODERATE", type="monomial")
    run_comparison(r=7, n=1800, p=6, label="HARD", type="monomial")

# %%