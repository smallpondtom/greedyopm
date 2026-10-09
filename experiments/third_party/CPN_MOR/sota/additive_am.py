import numpy as np
import scipy.optimize as opt
from pathlib import Path
import sys

sys.path.append("Adaptive_approach")
from train_utils.data_loader import myloader


def run(config):
    def kronecker_power(vec, p):
        # Start with the original vector
        kronecker_products = {tuple(vec)}
        # Compute the Kronecker product iteratively up to the power of p
        for _ in range(1, p):
            new_products = set()
            for prod in kronecker_products:
                new_prod = np.kron(prod, vec)
                new_products.add(tuple(new_prod))
            kronecker_products = new_products
        return [np.array(prod) for prod in kronecker_products]

    def representation_learning_obj(x):
        """Objective function for the nonlinear regression problem in the alternating minimization approach."""
        return S[:, snapshot] - sref - (V @ x) - (Vbar @ Xi @ polynomial_form(x))

    def polynomial_form(x):
        if len(x.shape) == 1:
            one_array = np.tile(1, 1)
            poly_g = [x ** degree for degree in range(1, p + 1)]
            Poly = np.concatenate(poly_g, axis=0)
            if use_kron_product:
                return np.concatenate([one_array, kronecker_power(x, p)])
            else:
                Poly = np.concatenate([one_array, Poly])
                return Poly
        else:
            one_array = np.tile(1, x.shape[1])
            poly_g = [x ** degree for degree in range(1, p + 1)]
            Poly = np.concatenate(poly_g, axis=0)
            if use_kron_product:
                W = [kronecker_power(x[:, i], p) for i in range(n_samples)]
                W = np.concatenate((np.expand_dims(one_array, 0), W), axis=0)
                return W
            else:
                Poly = np.concatenate((np.expand_dims(one_array, 0), Poly), axis=0)
                return Poly

    def relative_error(S, S_approx):
        return np.linalg.norm(S - S_approx, ord='fro') / np.linalg.norm(S, ord='fro')

    p = config["params"]["p"]
    N = config["params"]["N"]
    n = config["params"]["n"]
    method = config["params"]["method"]

    results_path = config["results_path"] + "/" + method
    folder = Path(results_path)
    folder.mkdir(parents=True, exist_ok=True)

    print("Method    =   ", method)
    recompute_svd = config["add_params"]["compute_svd"]
    path_svd = config["path_svd"]

    S, S_test, Sref, Sref_test = myloader(config)
    tol = 1e-4  # tolerence for alternating minimization
    gamma = 1e-4  # regularization parameter
    size_g = p * n + 1  # (p-1) * r or p * r +1
    max_iter = 50  # maximum number of iterations
    n_samples = Sref.shape[1]
    use_kron_product = False
    sref = Sref[:, 0]
    if use_kron_product:
        size_g = (size_g * (size_g - 1)) // 2 + 1

    ################################################################
    #  POD based learning
    ################################################################
    path_rob = Path(path_svd)
    if path_rob.exists() and recompute_svd == False:
        U = np.load(path_svd)
    else:
        U, ss, _ = np.linalg.svd(S - Sref, full_matrices=False)
    V = U[:, :n]
    Vbar = U[:, n:N]
    print("V shape    = ", V.shape)
    print("Vbar shape    = ", Vbar.shape)
    An = V.T @ (S - Sref)

    Proj_error = S - Sref - (V @ An)
    Poly = polynomial_form(An)
    Xi = Vbar.T @ Proj_error @ Poly.T @ np.linalg.inv(Poly @ Poly.T + gamma * np.identity(size_g))

    Gamma_MPOD = Sref + (V @ An) + (Vbar @ Xi @ Poly)
    print(f"\n POD based Reconstruction training error =  {relative_error(S, Gamma_MPOD):}")

    An_test = V.T @ (S_test - Sref_test)
    Poly_test = polynomial_form(An_test)
    Gamma_MPOD_test = Sref_test + (V @ An_test) + (Vbar @ Xi @ Poly_test)

    print(f"\n POD based Reconstruction test error =  {relative_error(S_test, Gamma_MPOD_test):}")
    np.save(results_path + "/test_additive_approx_without_AM.npy", Gamma_MPOD_test)

    ################################################################
    #  AM based learning
    ################################################################

    nrg_old = 0
    print("***Starting alternating minimizations:")

    # start iterations
    for niter in range(max_iter):

        # step 1 - orthogonal Procrustes (update basis vectors)
        U, _, VT = np.linalg.svd((S - Sref) @ np.concatenate([An, Xi @ Poly]).T, full_matrices=False)
        Omega = U @ VT

        V, Vbar = Omega[:, :n], Omega[:, n:N]

        # step 2 - linear regression (update coefficient matrix)
        Proj_error = S - Sref - (V @ An)
        rhs = np.linalg.inv(Poly @ Poly.T + (gamma * np.identity(size_g)))
        Xi = Vbar.T @ Proj_error @ Poly.T @ rhs

        # step 3 - nonlinear regression (update reduced state representation)
        for snapshot in range(n_samples):
            An[:, snapshot] = opt.least_squares(representation_learning_obj, An[:, snapshot], ftol=1e-7).x
        Poly = polynomial_form(An)

        # evaluate convergence criterion
        energy = np.linalg.norm(V @ An + (Vbar @ Xi @ Poly), 'fro') ** 2 / np.linalg.norm(S - Sref, 'fro') ** 2
        diff = abs(energy - nrg_old)
        print(f"\titeration: {niter + 1:d}\tsnapshot energy: {energy:e}\t diff: {diff:e}")
        if diff < tol:
            print("***Convergence criterion active!")
            break
        nrg_old = energy  # update old energy metric

    Gamma_MAM = Sref + (V @ An) + (Vbar @ Xi @ Poly)
    print(f"\n AM based Reconstruction training error =  {relative_error(S, Gamma_MAM):}")

    a_test = V.T @ (S_test - Sref_test)
    Poly_test = polynomial_form(a_test)
    Gamma_MAM_test = Sref_test + (V @ a_test) + (Vbar @ Xi @ Poly_test)
    print(f"\n AM based Reconstruction test error =  {relative_error(S_test, Gamma_MAM_test):}")

    np.save(results_path + "/test_additive_AM.npy", Gamma_MAM_test)
