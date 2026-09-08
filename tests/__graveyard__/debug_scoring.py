"""At iteration i after J = [0], verify that the scoring of GreedyOPM
(with gamma_sp -> 0, so all features active) matches GreedyFixedOPM's
argmin.  They should select the SAME j_star."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

from experiments.__old__.data.transport import generate_advecting_pulse
from greedyopm.polynomials.arnoldi import arnoldi_build, arnoldi_build_batch
from greedyopm.polynomials.multindex import graded_lex_multi_indices_for_degrees


if __name__ == "__main__":
    S, _, _ = generate_advecting_pulse(
        pulse_width=5e-4, pulse_shift=0.1, speed=5.0,
        final_time=0.20, n_time_samples=400, n_space_samples=256,
    )
    S_train = S[:, ::2]
    d, n = S_train.shape
    s_ref = jnp.mean(S_train, axis=1, keepdims=True)
    S_tilde = S_train - s_ref

    # SVD
    U_full, sigma_full, Wt_full = jnp.linalg.svd(S_tilde, full_matrices=False)
    s_eff = 30
    U_s = U_full[:, :s_eff]
    sigma_s = sigma_full[:s_eff]
    W_s = Wt_full[:s_eff, :].T
    SigmaWt = sigma_s[:, None] * W_s.T

    # Assume iter 0 picked mode 0. At iter 1 we have J=[0], complement = [1..s_eff-1].
    J = [0]
    complement = list(range(1, s_eff))
    cands_arr = jnp.array(complement, dtype=jnp.int32)
    n_cand = len(complement)

    print(f"n_cand = {n_cand}, n_snapshots = {n}")

    # ---- Build trial S_hat batch ----
    S_hat_old = SigmaWt[jnp.asarray(J)]  # (1, n)
    cand_row = SigmaWt[cands_arr][:, None, :]  # (n_cand, 1, n)
    trial_S_hat_all = jnp.concatenate(
        [jnp.broadcast_to(S_hat_old[None], (n_cand, 1, n)), cand_row],
        axis=1,
    )  # (n_cand, 2, n)

    mi_np = np.asarray(graded_lex_multi_indices_for_degrees(2, [2]), dtype=np.int32)
    Psi_all, _, _, _ = arnoldi_build_batch(trial_S_hat_all, mi_np, n_reorth=1)
    # (n_cand, 6, n)

    nl_start = 1 + 2  # = 3
    P_nl_all = Psi_all[:, nl_start:, :]  # (n_cand, 3, n)

    # ---- MY (greedy.py) scoring ----
    w_P_nl_all = jnp.einsum("nl,cmn->clm", W_s, P_nl_all)  # (n_cand, s, m_nl)
    w_P_at_cands = w_P_nl_all[:, cands_arr, :]              # (n_cand, n_cand, m_nl)
    sigma_cands = sigma_s[cands_arr]
    inv_n = 1.0 / n
    scaled = (inv_n * sigma_cands)[None, :, None] * w_P_at_cands
    sq = scaled ** 2
    total = jnp.sum(sq, axis=1)
    diag_ = jnp.diagonal(sq, axis1=0, axis2=1).T
    norms_sq = total - diag_
    poly_gain_sparse = n * jnp.sum(norms_sq, axis=1)
    delta_sparse = sigma_cands ** 2 + poly_gain_sparse

    # ---- Fixed-style scoring ----
    # For each c: error(c) = sum_{l in out(c)} sigma_l^2 - (1/n) * ||[SigmaWt[out(c)]] P_nl[c]^T||_F^2
    # We'll compute it with gamma=0 so error is -(1/n) * nl_correction up to constant.
    # Minimizing error <=> maximizing sigma_c^2 + (1/n) * nl_correction_all_features
    # where nl_correction_all_features = sum_{k, l in out(c)} (sigma_l W_s[:, l]^T P_nl[c, k])^2

    # For trial c with cand=cands[c], out(c) = {cands[0], ..., cands[c-1], cands[c+1], ..., cands[-1]}
    # nl_correction[c] = sum_{c' != c, k} (SigmaWt[cands[c'], :] · P_nl[c, k, :])^2
    # Which is EXACTLY what my sq computes. So let me aggregate along c', k axes:
    sum_all = jnp.sum(sq, axis=(1, 2))  # sum over c' and k → (n_cand,)
    sum_diag = jnp.sum(jnp.diagonal(sq, axis1=0, axis2=1), axis=0)  # (n_cand,)  (sum over k at c' = c)
    nl_correction = sum_all - sum_diag  # (n_cand,)
    # The (1/n)^2 from scaled ** 2 is already baked in. So my nl_correction has
    # the factor 1/n^2. The "fixed" formula has 1/n for nl_correction in the error.
    # To match: fixed uses (1/n) * sum_{c', k} (SigmaWt · P_nl)^2.  My scaled^2 is (1/n)^2 * same.
    # So: fixed_correction = n * my_nl_correction.
    fixed_correction = n * nl_correction
    delta_fixed_style = sigma_cands ** 2 + fixed_correction

    print("\n--- Comparison at iter=1 (after picking mode 0) ---")
    print(f"{'cand':>5s}  {'sigma^2':>12s}  {'poly_sparse':>14s}  {'poly_fixed':>14s}  "
          f"{'Δ_sparse':>14s}  {'Δ_fixed':>14s}")
    for c_idx in range(min(10, n_cand)):
        c = complement[c_idx]
        print(f"{c:5d}  {float(sigma_cands[c_idx]**2):12.4e}  "
              f"{float(poly_gain_sparse[c_idx]):14.4e}  {float(fixed_correction[c_idx]):14.4e}  "
              f"{float(delta_sparse[c_idx]):14.4e}  {float(delta_fixed_style[c_idx]):14.4e}")

    print()
    sparse_argmax = int(jnp.argmax(delta_sparse))
    fixed_argmax = int(jnp.argmax(delta_fixed_style))
    print(f"Sparse picks  cand idx {sparse_argmax} (mode {complement[sparse_argmax]})")
    print(f"Fixed picks   cand idx {fixed_argmax} (mode {complement[fixed_argmax]})")

    # Also: what's the DIFFERENCE between poly_sparse and poly_fixed?
    # poly_sparse = n * sum_{c' != c, k} (1/n)^2 (sigma_l W_s[l]^T P_nl[c,k])^2
    #             = (1/n) * sum_{c' != c, k} (sigma_l W_s[l]^T P_nl[c,k])^2
    # poly_fixed  = n * (1/n^2) * sum_{c' != c, k} (sigma_l W_s[l]^T P_nl[c,k])^2  [what I computed]
    # Wait — let me recompute. scaled = (1/n) * sigma_c' * w_P. scaled^2 = (1/n^2) * ...
    # norms_sq = sum over c' != c of scaled^2 = (1/n^2) sum_{c' != c, for one k} ...
    # poly_sparse = n * sum_k norms_sq = n * (1/n^2) * sum_{c' != c, k} ... = (1/n) * sum_{c', k}
    # nl_correction = sum over k of sum over c' != c of scaled^2 = same (1/n^2) * sum_{...}
    # fixed_correction = n * nl_correction = (1/n) * sum_{c', k}
    # So poly_sparse == fixed_correction. They should be IDENTICAL.

    print("\n|| poly_sparse - fixed_correction ||_inf =",
          float(jnp.max(jnp.abs(poly_gain_sparse - fixed_correction))))

    # So the scoring should produce the same ranking. Let's see if that's true:
    order_sparse = jnp.argsort(-delta_sparse)
    order_fixed = jnp.argsort(-delta_fixed_style)
    print(f"\nTop-10 sparse order: {[complement[int(i)] for i in order_sparse[:10]]}")
    print(f"Top-10 fixed  order: {[complement[int(i)] for i in order_fixed[:10]]}")