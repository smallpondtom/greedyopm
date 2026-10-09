import numpy as np
import tensap
import sys
from pathlib import Path
from joblib import Parallel, delayed
from timeit import default_timer
from train_utils.data_loader import myloader


def relative_error(S, S_approx):
    return np.linalg.norm(S - S_approx, ord='fro') / np.linalg.norm(S, ord='fro')


def tensor_solver(SOLVER, input_data, output_data):
    SOLVER.training_data = [None, output_data]
    SOLVER.test_data = [input_data, output_data]

    F, _ = SOLVER.solve()
    return F


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
        np.save(results_path + "/basis_vectors.npy", U)
    Vn = U[:, :n]
    Vbar = U[:, n:N]
    print("Vn shape    = ", Vn.shape)
    print("Vbar shape    = ", Vbar.shape)
    An = Vn.T @ (S - Sref)  # represent data in POD coordinates

    X = [tensap.UniformRandomVariable(np.min(x), np.max(x)) for x in An]
    BASIS = [
        tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(), range(p + 1))
        for x in X
    ]
    BASES = tensap.FunctionalBases(BASIS)
    SOLVER = tensap.TreeBasedTensorLearning.tensor_train_tucker(
        n, tensap.SquareLossFunction()
    )

    SOLVER.bases = BASES
    SOLVER.bases_eval = BASES.eval(An.T)

    SOLVER.tolerance["on_stagnation"] = 1e-6

    SOLVER.initialization_type = "canonical"

    SOLVER.linear_model_learning.regularization = False
    SOLVER.linear_model_learning.basis_adaptation = True
    SOLVER.linear_model_learning.error_estimation = True

    SOLVER.test_error = True
    # SOLVER.bases_eval_test = BASES.eval(X_TEST)

    SOLVER.rank_adaptation = True
    SOLVER.rank_adaptation_options["max_iterations"] = 20
    SOLVER.rank_adaptation_options["theta"] = 0.8
    SOLVER.rank_adaptation_options["early_stopping"] = True
    SOLVER.rank_adaptation_options["early_stopping_factor"] = 10

    SOLVER.tree_adaptation = False
    SOLVER.tree_adaptation_options["max_iterations"] = 1e2

    SOLVER.alternating_minimization_parameters["stagnation"] = 1e-10
    SOLVER.alternating_minimization_parameters["max_iterations"] = 50

    SOLVER.display = True
    SOLVER.alternating_minimization_parameters["display"] = False

    SOLVER.model_selection = True
    SOLVER.model_selection_options["type"] = "test_error"
    SOLVER.tolerance["on_error"] = 1e-4

    Abar = Vbar.T @ (S - Sref)
    t1 = default_timer()

    f = Parallel(n_jobs=-1)(
                delayed(tensor_solver)(SOLVER, An.T, Abar.T[:, i]) for i in range(Abar.T.shape[1]))
    Qbar_approx = np.array([f[i](An.T) for i in range(len(f))])

    S_approx = Sref + Vn @ An + Vbar @ Qbar_approx

    print(f"\n Tensor learning training error =  {relative_error(S, S_approx):}")

    An_test = Vn.T @ (S_test - Sref_test)
    Abar_approx_test = np.array([f[i](An_test.T) for i in range(len(f))])
    S_approx_test = Sref_test + Vn @ An_test + Vbar @ Abar_approx_test

    print(f"\n Tensor learning test error =  {relative_error(S_test, S_approx_test):}")
    np.save(results_path + "/coeffs_low_rank_approx.npy", Abar_approx_test)
    np.save(results_path + "/test_low_rank_approx.npy", S_approx_test)

    t2 = default_timer()
    print("time   =  ", t2 - t1)

