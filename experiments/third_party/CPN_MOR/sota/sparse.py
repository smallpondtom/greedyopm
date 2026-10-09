import numpy as np
import tensap
import sys
from pathlib import Path
from joblib import Parallel, delayed
from train_utils.data_loader import myloader
from train_utils.solve_ls import *


def relative_error(S, S_approx):
    return np.linalg.norm(S - S_approx, ord='fro') / np.linalg.norm(S, ord='fro')


def sparse_solver(s, input_data, output_data, A, H, ls):
    f = s.leastSquares(input_data, output_data, A, H, ls)

    return f


def run(config):
    p = config["params"]["p"]
    N = config["params"]["N"]
    n = config["params"]["n"]
    method = config["params"]["method"]

    print("Method    =   ", method)
    recompute_svd = config["add_params"]["compute_svd"]
    path_svd = config["path_svd"]

    S, S_test, Sref, Sref_test = myloader(config)
    results_path = config["results_path"] + "/" + method
    folder = Path(results_path)
    folder.mkdir(parents=True, exist_ok=True)
    path_rob = Path(path_svd)
    if path_rob.exists() and recompute_svd == False:
        U = np.load(path_svd)
    else:
        U, _, _ = np.linalg.svd(S - Sref, full_matrices=False)
        np.save(results_path + "/left_rob.npy", U)
    Vn = U[:, :n]
    Vbar = U[:, n:N]
    print("V shape    = ", Vn.shape)
    print("Vbar shape    = ", Vbar.shape)
    An = Vn.T @ (S - Sref)  # represent data in POD coordinates

    S_POD = Sref + Vn @ An  # compute the projection of the original dataset
    print(f"\n Traditional POD Reconstruction error =  {relative_error(S, S_POD):}")

    ls = tensap.LinearModelLearningSquareLoss()
    ls.regularization = True
    ls.regularization_type = "l1"
    ls.regularization_options = {"alpha": 0.}
    ls.model_selection = True
    ls.error_estimation = True
    ls.error_estimation_type = 'leave_out'
    ls.error_estimation_options["correction"] = True

    X = [tensap.UniformRandomVariable(np.min(x), np.max(x)) for x in An]
    BASIS = [
        tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(), range(p + 1))
        for x in X
    ]
    BASES = tensap.FunctionalBases(BASIS)
    d = An.shape[0]
    I = tensap.MultiIndices.hyperbolic_cross_set(d, p)

    s = solve_ls(dim=d, I=I, maxIndex=p)
    H = tensap.SparseTensorProductFunctionalBasis(BASES, I)
    A = H.eval(An.T)
    Abar = Vbar.T @ (S - Sref)
    f = Parallel(n_jobs=-1)(
        delayed(sparse_solver)(s, An.T, Abar.T[:, i], A, H, ls) for i in range(Abar.T.shape[1]))
    # f = s.leastSquares(An.T, Abar.T, A, H, ls)
    Abar_approx = np.array([f[i](An.T) for i in range(len(f))])

    S_approx = Sref + Vn @ An + Vbar @ Abar_approx

    print(f"\n Sparse training error =  {relative_error(S, S_approx):}")

    An_test = Vn.T @ (S_test - Sref_test)
    Abar_approx_test = np.array([f[i](An_test.T) for i in range(len(f))])
    S_approx_test = Sref_test + Vn @ An_test + Vbar @ Abar_approx_test

    print(f"\n Sparse test error =  {relative_error(S_test, S_approx_test):}")

    np.save(results_path + "/coeffs_sparse.npy", Abar_approx_test)
    np.save(results_path + "/test_sparse_approx.npy", S_approx_test)
