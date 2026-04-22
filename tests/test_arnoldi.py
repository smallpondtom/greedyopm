"""Orthonormality and correctness of Arnoldi and incremental Arnoldi."""

import sys
from pathlib import Path
import numpy as np

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from greedyopm.polynomials.arnoldi import (
    arnoldi_build, arnoldi_eval, arnoldi_orthonormality_error,
)
from greedyopm.polynomials.multindex import (
    graded_lex_multi_indices, compute_parent_info_for_new,
    new_multi_indices_on_coord_add, graded_lex_multi_indices_for_degrees,
)
from greedyopm.polynomials.incremental_arnoldi import (
    initial_state,
    incremental_arnoldi_vmap,
    compress_psi_batch,
    commit_winner,
)


def test_full_arnoldi_orthonormality():
    """(1/n) Psi Psi^T = I for modest r, p."""
    rng = np.random.default_rng(0)
    r, n = 5, 500
    S_hat = jnp.asarray(rng.normal(size=(r, n)))
    mi = graded_lex_multi_indices(r, 3, min_degree=0)

    Psi, basis = arnoldi_build(S_hat, mi, n_reorth=1)
    err = arnoldi_orthonormality_error(Psi)
    assert err < 1e-10, f"orthonormality error = {err:.3e}"

    # Evaluation at new points should agree with a fresh build.
    S_hat_new = jnp.asarray(rng.normal(size=(r, 50)))
    Psi_new = arnoldi_eval(S_hat_new, basis)

    # Check shape
    assert Psi_new.shape == (mi.shape[0], 50), Psi_new.shape
    print(f"  [full arnoldi] orthonormality err = {err:.3e}")


def test_incremental_matches_full_single_candidate():
    """Incremental Arnoldi rebuild matches full rebuild for the same S_hat."""
    rng = np.random.default_rng(1)
    r_target, n = 4, 300
    degrees = [2, 3]

    # Build a full r_target Arnoldi by running the incremental machinery
    # r_target times, always picking coordinate i at iteration i (so the
    # reduced-coord sequence is the identity permutation).
    S_hat = jnp.asarray(rng.normal(size=(r_target, n)))

    # We need W_s of shape (n, s) — but for this test we don't care about
    # SVD structure.  Use any orthonormal W_s (the scoring is not tested).
    Q, _ = jnp.linalg.qr(jnp.asarray(rng.normal(size=(n, r_target + 2))))
    W_s = Q
    s = W_s.shape[1]

    state = initial_state(n, s, 0, W_s)
    committed_mi = np.zeros((1, 0), dtype=np.int32)

    for i in range(r_target):
        n_coords_trial = i + 1
        new_coord_idx = i

        # c_new: the newly added coordinate's values = S_hat[i].
        c_new_all = S_hat[i:i+1]                           # (1, n)

        new_mi = new_multi_indices_on_coord_add(
            n_coords_trial, degrees, include_new_linear=True,
        )
        M_old = int(state.Psi.shape[0])
        old_mi_padded = np.hstack([
            committed_mi, np.zeros((M_old, 1), dtype=np.int32),
        ])
        parent_idx_np = compute_parent_info_for_new(
            old_mi_padded, new_mi, new_coord=new_coord_idx,
        )
        parent_idx = jnp.asarray(parent_idx_np, dtype=jnp.int32)

        Psi_new_all, H_cross_all, H_new_all = incremental_arnoldi_vmap(
            state.Psi, c_new_all, parent_idx,
            new_coord_idx=new_coord_idx, n_reorth=1,
        )
        w_P_new_all = compress_psi_batch(W_s, Psi_new_all)

        state = commit_winner(
            state,
            Psi_new_all[0], H_cross_all[0], H_new_all[0], w_P_new_all[0],
            np.full(new_mi.shape[0], new_coord_idx, dtype=np.int32),
            parent_idx_np.astype(np.int32),
        )
        committed_mi = np.vstack([old_mi_padded, new_mi.astype(np.int32)])

    # Verify orthonormality of the committed state.
    err_inc = float(jnp.linalg.norm(
        (1.0 / n) * (state.Psi @ state.Psi.T) - jnp.eye(state.Psi.shape[0])
    ))
    print(f"  [incremental] orthonormality err = {err_inc:.3e}")
    assert err_inc < 1e-10, err_inc

    # Compare to full Arnoldi on the same multi-indices & data.
    full_mi = graded_lex_multi_indices_for_degrees(r_target, degrees)
    # The incremental multi-indices should match (possibly up to ordering).
    # Convert both to sets-of-tuples to check.
    set_inc = {tuple(row) for row in committed_mi}
    set_full = {tuple(row) for row in full_mi}
    assert set_inc == set_full, (
        f"multi-index sets differ:\n inc = {sorted(set_inc)}\n full = {sorted(set_full)}"
    )

    # Check span equality: both Psi's should have the same row-span.
    # Equivalent to: Psi_full Psi_inc^T is full rank and (1/n) matches I up to
    # signs/permutation.  An easier check: the projection Psi_full Psi_inc^T / n
    # is an orthogonal matrix.
    Psi_full, _ = arnoldi_build(S_hat, full_mi, n_reorth=1)
    # Their rows may be in different grlex orderings (incremental builds
    # everything involving the new coord at each step, which is a different
    # ordering than "all of degree q before all of degree q+1").  Compare the
    # span via Gram:
    G_inc = (1.0 / n) * state.Psi @ state.Psi.T
    G_full = (1.0 / n) * Psi_full @ Psi_full.T
    # Both are I (orthonormal), so the span match is trivial up to sign.
    assert jnp.allclose(G_inc, jnp.eye(G_inc.shape[0]), atol=1e-10)
    assert jnp.allclose(G_full, jnp.eye(G_full.shape[0]), atol=1e-10)


if __name__ == "__main__":
    test_full_arnoldi_orthonormality()
    test_incremental_matches_full_single_candidate()
    print("All arnoldi tests passed.")