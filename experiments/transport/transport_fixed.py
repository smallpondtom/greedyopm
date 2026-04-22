"""
Compare Greedy-Fixed-OPM vs Greedy-QM on the linear transport (advecting pulse) 
problem.
"""

#%% ---------------------------------------------------------------------------
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

#%% ---------------------------------------------------------------------------
# Make experiment modules importable
# -----------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "experiments"))

from data.transport import generate_advecting_pulse
from qm.quadmani import quadmani_greedy, quadratic_mapping, cubic_mapping
from greedyopm import GreedyFixedOPM


#%% --------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def relative_error(S_true, S_approx):
    """Relative Frobenius error  ||S - S_approx||_F / ||S||_F."""
    return float(jnp.linalg.norm(S_true - S_approx, "fro")
                 / jnp.linalg.norm(S_true, "fro"))


def pod_reconstruct(S_train, S_test, r, center=True):
    """Baseline: rank-r linear POD reconstruction of test data."""
    if center:
        s_bar = jnp.mean(S_train, axis=1, keepdims=True)
    else:
        s_bar = jnp.zeros((S_train.shape[0], 1))
    U, _, _ = jnp.linalg.svd(S_train - s_bar, full_matrices=False)
    U_r = U[:, :r]
    S_test_c = S_test - s_bar
    return s_bar + U_r @ (U_r.T @ S_test_c)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

#%%
if __name__ == "__main__":
    # --- Setup ---------------------------------------------------------------
    r = 10
    s = 100
    p = 2 

    #! Experiment 1: Running each method once on training data

    #%% --- Generate data -----------------------------------------------------
    print("\n[1/5] Generating advecting pulse data ...")
    S, xspan, tspan = generate_advecting_pulse(
        pulse_width=5.0e-4,
        pulse_shift=0.1,
        speed=5.0,
        final_time=0.20,
        n_time_samples=2000,
        n_space_samples=2**10,
    )
    n, k = S.shape
    print("     Snapshot matrix: %d x %d  (n_space x n_time)" % (n, k))

    # Train/test split: even columns for training, odd for testing
    S_train = S[:, ::2]
    S_test = S[:, 1::2]
    k_train = S_train.shape[1]
    k_test = S_test.shape[1]
    print("     Train: %d snapshots,  Test: %d snapshots" % (k_train, k_test))

    #%% --- Linear POD baseline -----------------------------------------------
    print("\n[2/5] Running POD baseline ...")
    s_ref = jnp.mean(S_train, axis=1, keepdims=True)
    V_pod, Sigma_pod, _ = jnp.linalg.svd(S_train - s_ref, full_matrices=False)

    # Training data
    S_pod_recon = s_ref + V_pod[:, :r] @ (V_pod[:, :r].T @ (S_train - s_ref))
    pod_err = relative_error(S_train, S_pod_recon)
    print("     POD reconstruction error (train) = %.6e" % pod_err)

    # Test data
    S_pod_test_recon = s_ref + V_pod[:, :r] @ (V_pod[:, :r].T @ (S_test - s_ref))
    pod_test_err = relative_error(S_test, S_pod_test_recon)
    print("     POD reconstruction error (test) = %.6e" % pod_test_err)

    #%% --- Greedy-QM ---------------------------------------------------------
    print("\n[3/5] Running Greedy-QM ...")
    V_qm, Vbar_qm, mu_qm, J_qm = quadmani_greedy(
        S_train, r, s, 1e-15, np.array([], dtype=int),
    )
    mu_qm = mu_qm.reshape(-1, 1) 

    # Training data
    Shat_qm = V_qm.T @ (S_train - mu_qm)
    S_qm_recon = mu_qm + V_qm @ (Shat_qm) + Vbar_qm @ quadratic_mapping(Shat_qm)
    qm_err = relative_error(S_train, S_qm_recon)
    print("     Selected indices: ", J_qm)
    print("     Greedy-QM reconstruction error (train) = %.6e" % qm_err)

    # Test data 
    Shat_test_qm = V_qm.T @ (S_test - mu_qm)
    S_qm_test_recon = mu_qm + V_qm @ (Shat_test_qm) + Vbar_qm @ quadratic_mapping(Shat_test_qm)
    qm_test_err = relative_error(S_test, S_qm_test_recon)
    print("     Greedy-QM reconstruction error (test) = %.6e" % qm_test_err)


    #%% --- Greedy-CM --------------------------------------------------------- 
    print("\n[4/5] Running Greedy-CM ...")
    V_cm, Vbar_cm, mu_cm, J_cm = quadmani_greedy(
        S_train, r, s, 1e-12, np.array([], dtype=int), 
        feature_map=cubic_mapping
    )
    mu_cm = mu_cm.reshape(-1,1)

    # Training data 
    Shat_cm = V_cm.T @ (S_train - mu_cm)
    S_cm_recon = mu_cm + V_cm @ (Shat_cm) + Vbar_cm @ cubic_mapping(Shat_cm)
    cm_err = relative_error(S_train, S_cm_recon)
    print("     Selected indices: ", J_cm)
    print("     Greedy-CM reconstruction error (train) = %.6e" % cm_err)

    # Test data 
    Shat_test_cm = V_cm.T @ (S_test - mu_cm)
    S_cm_test_recon = mu_cm + V_cm @ (Shat_test_cm) + Vbar_cm @ cubic_mapping(Shat_test_cm)
    cm_test_err = relative_error(S_test, S_cm_test_recon)
    print("     Greedy-CM reconstruction error (test) = %.6e" % cm_test_err)
    

    #%% --- Greedy-Fixed-OPM --------------------------------------------------
    print("\n[5/5] Running Greedy-Fixed-OPM ...")
    opm_configs = [
        {"family": "monomial",  "label": "OPM-Monomial"},
        {"family": "legendre",  "label": "OPM-Legendre"},
        {"family": "chebyshev", "label": "OPM-Chebyshev"},
        {"family": "hermite",   "label": "OPM-Hermite"},
    ]
    opm_config = opm_configs[1]
    gopm = GreedyFixedOPM(
        r=r,
        s=s,
        p=[3],
        gamma=1e-12,
        n_reorth=2,
        R_rank_tol=1e-9,
        family=opm_config["family"],
        ortho_method="qr",
        center=True,
    )
    gopm.fit(S_train)

    #%% Reconstruct training data
    S_opm_recon = gopm.reconstruct(S_train)
    opm_err = relative_error(S_train, S_opm_recon)
    print("     Selected indices: ", gopm.selected_modes)
    print("     Greedy-OPM reconstruction error (train) = %.6e" % opm_err)
    gopm.summary()

    #%% Reconstruct test data
    S_opm_test_recon = gopm.reconstruct(S_test)
    opm_test_err = relative_error(S_test, S_opm_test_recon)
    print("     Greedy-OPM reconstruction error (test) = %.6e" % opm_test_err)

# %%
