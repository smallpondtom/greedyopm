"""
Validation: multivariate Arnoldi vs QR-on-monomials.

In exact arithmetic, both algorithms produce orthonormal bases for the same
polynomial space — they differ only by a signed permutation/rotation.  In
finite precision they diverge as (r, p) grow: the QR route loses orthogonality
and, more importantly, R becomes so ill-conditioned that out-of-sample
evaluation via R^{-T} @ Phi_new amplifies round-off.

Four independent tests are run in three regimes:

  T1  Training orthogonality:     ||(1/n) Psi Psi^T - I||_F.
      Sanity check per method.

  T2  Common span:                ||Psi_arn - U @ Psi_qr||_F, where
      U = (1/n) Psi_arn @ Psi_qr^T.  In exact arithmetic U is orthogonal
      and this residual is zero — it certifies both methods span the same
      polynomial subspace.

  T3  Monomial reproduction:      project each monomial onto each basis
      and back.  Since monomials lie in the span, both methods must
      reconstruct them exactly.  A robust test that fails loudly when
      orthogonality is lost.

  T4  Out-of-sample agreement:    fit a known polynomial at training
      points, evaluate at held-out points.  Compares (i) QR vs truth,
      (ii) Arnoldi vs truth, (iii) QR vs Arnoldi.  Isolates the R^{-T}
      amplification failure mode of the QR pipeline.
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

jax.config.update('jax_enable_x64', True)

# ---------------------------------------------------------------------------
# One comparison at fixed (r, n, p)
# ---------------------------------------------------------------------------

def run_comparison(r: int, n: int, p: int, *, seed: int = 0, label: str = ""):
    print(f"\n{'='*72}")
    print(f"  {label}    r={r},  n={n},  p={p}")
    print(f"{'='*72}")

    rng = np.random.default_rng(seed)

    S, _, _ = generate_advecting_pulse(
        pulse_width=5.0e-4,
        pulse_shift=0.1,
        speed=5.0,
        final_time=0.20,
        n_time_samples=2000,
        n_space_samples=2**10,
    )

    # Center data and scale to [0,1]
    S_centered = S - jnp.mean(S, axis=1, keepdims=True)
    S_max = jnp.max(jnp.abs(S_centered))
    S_min = jnp.min(S_centered)
    S = (S_centered - S_min) / (S_max - S_min)

    V = jax.lax.linalg.svd(S, full_matrices=False)[0]    
    Vr = V[:, :r]    
    S_hat = Vr.T @ S
    S_hat = S_hat[:, :n]

    # S_hat = jnp.asarray(rng.standard_normal((r, n)))

    mi_np = graded_lex_multi_indices(r, p, min_degree=0)
    mi_j = jnp.asarray(mi_np)
    m = mi_np.shape[0]
    print(f"  total multi-indices m = {m}")

    # --- build both bases ---------------------------------------------------
    # Phi = evaluate_monomials(S_hat, mi_j)                 # (m, n)
    Phi = evaluate_classical_tensor(
                S_hat, mi_j, "legendre", p,
    )
    Psi_qr, R = orthogonalize_qr_refined(Phi, 1)          # (m, n), (m, m)
    Psi_arn, basis = arnoldi_build(S_hat, mi_np, n_reorth=1)

    # Report the R-diagonal spread — this is the conditioning indicator
    diag_R = jnp.abs(jnp.diag(R))
    diag_H = jnp.abs(jnp.diag(basis.H))
    print(f"  R  diagonal spread : min={float(jnp.min(diag_R)):.2e}  "
          f"max={float(jnp.max(diag_R)):.2e}")
    print(f"  H  diagonal spread : min={float(jnp.min(diag_H)):.2e}  "
          f"max={float(jnp.max(diag_H)):.2e}")

    # --- T1: training orthogonality ----------------------------------------
    I_m = jnp.eye(m)
    err_qr_ortho  = float(jnp.linalg.norm((1/n) * Psi_qr  @ Psi_qr.T  - I_m))
    err_arn_ortho = float(jnp.linalg.norm((1/n) * Psi_arn @ Psi_arn.T - I_m))
    print(f"  T1 ortho error QR        : {err_qr_ortho:.3e}")
    print(f"  T1 ortho error Arnoldi   : {err_arn_ortho:.3e}")

    # --- T2: common span ---------------------------------------------------
    # U = (1/n) Psi_arn Psi_qr^T.  If both span the same subspace and both
    # are orthonormal, U is an orthogonal m x m matrix, and Psi_arn = U Psi_qr.
    U = (1.0 / n) * (Psi_arn @ Psi_qr.T)
    span_residual = float(jnp.linalg.norm(Psi_arn - U @ Psi_qr))
    unitary_gap   = float(jnp.linalg.norm(U @ U.T - I_m))
    print(f"  T2 ||Psi_arn - U Psi_qr||: {span_residual:.3e}")
    print(f"  T2 ||U U^T - I||         : {unitary_gap:.3e}")

    # --- T3: monomial reproduction -----------------------------------------
    # Every monomial (row of Phi) lies in the span of each basis.  Project
    # onto the basis and reconstruct; residual should be ~0.
    Phi_fro = float(jnp.linalg.norm(Phi))
    proj_qr   = (Phi @ Psi_qr.T  / n) @ Psi_qr
    proj_arn  = (Phi @ Psi_arn.T / n) @ Psi_arn
    recon_qr  = float(jnp.linalg.norm(proj_qr  - Phi)) / Phi_fro
    recon_arn = float(jnp.linalg.norm(proj_arn - Phi)) / Phi_fro
    print(f"  T3 monomial recon QR     : {recon_qr:.3e}")
    print(f"  T3 monomial recon Arnoldi: {recon_arn:.3e}")

    # --- T4: out-of-sample agreement ---------------------------------------
    n_test = 200
    S_test = jnp.asarray(rng.standard_normal((r, n_test)))
    Phi_test = evaluate_monomials(S_test, mi_j)           # (m, n_test)

    # Target: a random polynomial in the span (so the truth is exactly
    # representable and we can measure deviation from it).
    a_true = jnp.asarray(rng.standard_normal(m))
    f_train      = a_true @ Phi                           # (n,)
    f_test_truth = a_true @ Phi_test                      # (n_test,)

    # QR pipeline: evaluate Psi at test points via the inverse R factor.
    R_inv_T = jnp.linalg.inv(R.T)
    Psi_qr_test = R_inv_T @ Phi_test                      # (m, n_test)
    c_qr = (1.0 / n) * (Psi_qr @ f_train)
    f_qr = c_qr @ Psi_qr_test

    # Arnoldi pipeline: evaluate Psi at test points via stored recurrence.
    Psi_arn_test = arnoldi_eval(S_test, basis)            # (m, n_test)
    c_arn = (1.0 / n) * (Psi_arn @ f_train)
    f_arn = c_arn @ Psi_arn_test

    truth_nrm = float(jnp.linalg.norm(f_test_truth))
    rel_qr    = float(jnp.linalg.norm(f_qr  - f_test_truth)) / truth_nrm
    rel_arn   = float(jnp.linalg.norm(f_arn - f_test_truth)) / truth_nrm
    cross     = float(jnp.linalg.norm(f_qr - f_arn)) / (
        jnp.linalg.norm(f_qr) + 1e-300
    )
    print(f"  T4 QR      vs truth      : {rel_qr:.3e}")
    print(f"  T4 Arnoldi vs truth      : {rel_arn:.3e}")
    print(f"  T4 QR      vs Arnoldi    : {cross:.3e}")


#%% -------------------------------------------------------------------------
# Three regimes spanning easy to pathological
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Easy: low degree, small r.  Everything agrees at ~1e-13.
    run_comparison(r=3, n=500, p=2, label="EASY")

    # Moderate: QR starts to show strain in R conditioning.
    run_comparison(r=5, n=800, p=3, label="MODERATE")

    # Hard: the R^{-T} inversion is numerically catastrophic for the QR
    # pipeline.  Arnoldi's recurrence-based evaluation should be unaffected.
    run_comparison(r=8, n=1200, p=4, label="HARD")