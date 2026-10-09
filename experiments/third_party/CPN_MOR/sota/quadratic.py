import numpy as np
import scipy.optimize as opt
from pathlib import Path
import sys

sys.path.append("Adaptive_approach")
from train_utils.data_loader import myloader


################################################################
#  Approximation error
################################################################
def relative_error(S, S_approx):
    return np.linalg.norm(S - S_approx, ord='fro') / np.linalg.norm(S, ord='fro')


def compute_kronecker(A, B):
    if A.shape != B.shape:
        raise ValueError("Both matrices should be of same shape")
    N, n = A.shape
    result = []
    for i in range(n):
        for j in range(i, n):
            prod = A[:, i] * B[:, j]
            result.append(prod)
    return np.column_stack(result)


def run(config):
    n = config["params"]["n"]
    method = config["params"]["method"]

    print("Method    =   ", method)
    recompute_svd = config["add_params"]["compute_svd"]
    path_svd = config["path_svd"]

    S, S_test, Sref, Sref_test = myloader(config)
    results_path = config["results_path"] + "/" + method
    folder = Path(results_path)
    folder.mkdir(parents=True, exist_ok=True)

    regs = np.logspace(-3, 3, 20)
    list_train_errors = []
    list_test_errors = []
    S_approx_list = np.zeros((S_test.shape[0], S_test.shape[1], len(regs)))
    S_centered = S - Sref
    path_rob = Path(path_svd)
    if path_rob.exists() and recompute_svd == False:
        U = np.load(path_svd)
    else:
        U, ss, _ = np.linalg.svd(S_centered, full_matrices=False)
        np.save(results_path + "/basis_vectors.npy", U)
    V = U[:, :n]
    An = V.T @ S_centered

    W = compute_kronecker(An.T, An.T).T
    print("W shape   =   ", W.shape)
    dim2 = W.shape[0]
    # Vbar = ((np.eye(dim) - V @ V.T) @ S_centered @ W.T @ np.linalg.inv(W @ W.T + reg * np.eye(dim2)))

    for i, reg in enumerate(regs):
        Vbar = S_centered @ W.T @ np.linalg.inv(W @ W.T + reg * np.eye(dim2)) - \
               V @ (V.T @ S_centered @ W.T @ np.linalg.inv(W @ W.T + reg * np.eye(dim2)))
        S_quad_approx = Sref + V @ An + Vbar @ W

        s_hat_test = V.T @ (S_test - Sref_test)
        W_test = compute_kronecker(s_hat_test.T, s_hat_test.T).T
        S_quad_approx_test = Sref_test + V @ s_hat_test + Vbar @ W_test
        list_train_errors.append(relative_error(S, S_quad_approx))
        list_test_errors.append(relative_error(S_test, S_quad_approx_test))
        S_approx_list[:, :, i] = S_quad_approx_test

    min_index = np.argmin(np.array(list_test_errors))

    print("Quadratic manifold training error =", list_train_errors[min_index])
    print("Quadratic manifold test error =", list_test_errors[min_index])
    np.save(results_path + "/test_quadratic_approx.npy", S_approx_list[:, :, min_index])
