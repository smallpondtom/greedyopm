import numpy as np
from pathlib import Path
import sys
import yaml
from argparse import ArgumentParser

sys.path.append("Adaptive_approach")
from train_utils.data_loader import myloader


################################################################
#  Approximation error
################################################################
def relative_error(S, S_approx):
    return np.linalg.norm(S - S_approx, ord='fro') / np.linalg.norm(S, ord='fro')


def run(config):

    n = config["params"]["n"]
    method = config["params"]["method"]
    recompute_svd = config["add_params"]["compute_svd"]
    path_svd = config["path_svd"]

    print("Method    =   ", method)
    S, S_test, Sref, Sref_test = myloader(config)
    results_path = config["results_path"] + "/" + method
    folder = Path(results_path)
    folder.mkdir(parents=True, exist_ok=True)
    path_rob = Path(path_svd)
    if path_rob.exists() and recompute_svd == False:
        U = np.load(path_svd)
    else:
        U, _, _ = np.linalg.svd(S - Sref, full_matrices=False)
        np.save(results_path + "/basis_vectors.npy", U)
    V = U[:, :n]
    An = V.T @ (S - Sref)
    An_test = V.T @ (S_test - Sref_test)
    S_approx = Sref + V @ An
    S_approx_test = Sref_test + V @ An_test

    print("Linear manifold training error = ", relative_error(S, S_approx))
    print("Linear manifold test error = ", relative_error(S_test, S_approx_test))

    np.save(results_path + "/test_linear_approx.npy", S_approx_test)

