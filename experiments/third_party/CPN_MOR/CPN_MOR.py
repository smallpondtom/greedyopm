import matplotlib.pyplot as plt
from train_utils.utilities_tensor_learning import CPN_LR
from train_utils.utilities_sparse import *

from train_utils.data_loader import myloader
from pathlib import Path
from timeit import default_timer
import yaml
from argparse import ArgumentParser
from visualization.tree_viz import tree_visualization
from encoder_decoder import Encoder, Decoder
import pickle


def relative_error(S, S_approx, setting="mean_squared"):
    if setting == "mean_squared":
        return np.linalg.norm(S - S_approx, 'fro') / np.linalg.norm(S, 'fro')
    else:
        return max(np.linalg.norm(S - S_approx, axis=0)) / max(np.linalg.norm(S, axis=0))


def error_svd(singular_values):
    error = np.sqrt(np.flip(np.cumsum(np.flip(singular_values ** 2)) / (singular_values ** 2).sum()))
    return error


def run():
    folder = Path(results_path)
    folder.mkdir(parents=True, exist_ok=True)
    if setting == "mean_squared":
        path_left_rob = config["path_svd"]
        path_rob = Path(path_left_rob)
        if path_rob.exists() and recompute_svd == False:
            U = np.load(path_left_rob)
        else:
            print("Computing SVD...")
            U, Sigma, _ = np.linalg.svd(S - Sref, full_matrices=False)
            plt.semilogy(range(1, len(error_svd(Sigma))), error_svd(Sigma)[1:], marker='o')
            plt.grid()
            plt.savefig(results_path + "/singular_value_decay.png")
        np.save(results_path + "/singular_vectors.npy", U)
        print("SVD truncation...")

        Vstar, Qstar = method.truncate_svd(U[:, ])
    elif setting == "worst_case":
        Vstar, Qstar, _, _ = method.greedy_snapshots(verbose=True)
    else:
        raise ValueError("Invalid setting value")

    print("V_N shape= ", Vstar.shape)
    print("A_N shape= ", Qstar.shape)

    t1 = default_timer()

    Gamma = config["add_params"]["L"]
    train_set = config["params"]["train_val_set"]
    tol_min, n_min = False, False
    if "n_min" in config["add_params"]:
        n_min = config["add_params"]["n_min"]
    else:
        tol_min = config["add_params"]["tol_min"]

    Qr, func, index_r, lipschitz_consts = method.find_n(Vstar, Qstar, p1=p1, tol_min=tol_min, n_min=n_min,
                                                        train_set=train_set, Gamma=Gamma)

    print("A_n shape = ", Qr.shape)
    indices = np.arange(Vstar.shape[1])
    n = len(index_r)
    V = Vstar[:, index_r]

    Vbar = np.delete(Vstar, index_r, axis=1)
    E = Encoder(index=np.array(index_r), basis=V, u_ref=sref)
    D = Decoder(linear_index=np.array(index_r), nonlinear_index=np.delete(indices, index_r),
                linear_basis=V, nonlinear_basis=Vbar, functions_f=func,
                u_ref=sref)

    Qbar = D.eval_all_coeffs(Qr)
    Qtest = Vstar.T @ (S_test - Sref_test)
    S_lin = Sref + Vstar[:, :n] @ Qstar[:n, :]
    S_lin_test = Sref_test + Vstar[:, :n] @ Qtest[:n, :]
    S_approx = Sref + V @ Qr + Vbar @ Qbar
    t2 = default_timer()

    Qr_test = V.T @ (S_test - Sref_test)

    Qbar_test = D.eval_all_coeffs(Qr_test)
    S_approx_test = Sref_test + V @ Qr_test + Vbar @ Qbar_test

    with open(results_path + "/Encoder.pkl", "wb") as f:
        pickle.dump(E, f)

    with open(results_path + "/Decoder.pkl", "wb") as f:
        pickle.dump(D, f)

    print("time = ", t2 - t1, " secs")

    print("Decoder lipschitz const = ", np.sqrt(1 + sum(np.array(lipschitz_consts) ** 2)))
    if setting == "worst_case":
        print("# ----------------- WORST-CASE SETTING ----------------- #")
        print("Linear reconstruction training error = ", relative_error(S, S_lin, setting="worst_case"))
        print("Linear reconstruction test error = ", relative_error(S_test, S_lin_test, setting="worst_case"))
        print("Nonlinear reconstruction training error = ", relative_error(S, S_approx, setting="worst_case"))
        print("Nonlinear reconstruction test error = ", relative_error(S_test, S_approx_test, setting="worst_case"))

    print("# ----------------- MEAN-SQUARED SETTING ----------------- #")
    print("Linear reconstruction training error = ", relative_error(S, S_lin, setting="mean_squared"))
    print("Linear reconstruction test error = ", relative_error(S_test, S_lin_test, setting="mean_squared"))
    print("Nonlinear reconstruction training error = ", relative_error(S, S_approx, setting="mean_squared"))
    print("Nonlinear reconstruction test error = ", relative_error(S_test, S_approx_test, setting="mean_squared"))
    tree_visualization(config)

    with open(Path(folder, "train_info.txt"), "w") as f:
        s = ""
        s += f"  * p          = {config['params']['p']:_}\n"
        s += f"  * tolerance          = {config['params']['tolerance']}\n"
        s += f"  * alpha          = {config['add_params']['alpha']:}\n"
        s += f"  * beta          = {config['add_params']['beta']:}\n"
        s += f"  * L          = {config['add_params']['L']:}\n"
        s += f"  * tensor learning          = {config['params']['approximation_type']:}\n"
        f.write(s)

    with open(Path(folder, "results_info.txt"), "w") as f:
        s = ""
        s += f"  * Time          = {t2 - t1:_}\n"
        s += f"  * n          = {len(index_r):_}\n"
        s += f"  * I          = {str(index_r)}\n"
        if setting == "worst_case":
            s += f"# ----------------- WORST-CASE SETTING ----------------- #\n"
            s += f"  * Linear reconstruction error training          = {relative_error(S, S_lin, setting='worst_case'):}\n"
            s += f"  * Linear reconstruction error test          = {relative_error(S_test, S_lin_test, setting='worst_case'):}\n"
            s += f"  * Nonlinear reconstruction error training          = {relative_error(S, S_approx, setting='worst_case'):}\n"
            s += f"  * Test reconstruction error          = {relative_error(S_test, S_approx_test, setting='worst_case'):}\n"
        s += f"# ----------------- MEAN-SQUARED SETTING ----------------- #\n"
        s += f"  * Linear reconstruction error training          = {relative_error(S, S_lin, setting='mean_squared'):}\n"
        s += f"  * Linear reconstruction error test          = {relative_error(S_test, S_lin_test, setting='mean_squared'):}\n"
        s += f"  * Nonlinear reconstruction error training          = {relative_error(S, S_approx, setting='mean_squared'):}\n"
        s += f"  * Test reconstruction error          = {relative_error(S_test, S_approx_test, setting='mean_squared'):}\n"
        s += f"  * Decoder lipschitz constant          = {np.sqrt(1 + sum(np.array(lipschitz_consts) ** 2)):}\n"
        f.write(s)


def test():
    folder = Path(results_path)
    if folder.exists():

        with open(results_path + "/Encoder.pkl", "rb") as f:
            E = pickle.load(f)

        with open(results_path + "/Decoder.pkl", "rb") as f:
            D = pickle.load(f)

        V = E.basis
        Qr = E(S)
        Qr_test = E(S_test)

        S_lin = Sref + V @ Qr
        S_lin_test = Sref_test + V @ Qr_test
        S_approx = D(Qr)

        S_approx_test = D(Qr_test)

        if setting == "worst_case":
            print("# ----------------- WORST-CASE SETTING ----------------- #")
            print("Linear reconstruction training error = ", relative_error(S, S_lin, setting="worst_case"))
            print("Linear reconstruction test error = ", relative_error(S_test, S_lin_test, setting="worst_case"))
            print("Nonlinear reconstruction training error = ", relative_error(S, S_approx, setting="worst_case"))
            print("Nonlinear reconstruction test error = ", relative_error(S_test, S_approx_test, setting="worst_case"))

        print("# ----------------- MEAN-SQUARED SETTING ----------------- #")
        print("Linear reconstruction training error = ", relative_error(S, S_lin, setting="mean_squared"))
        print("Linear reconstruction training error = ", relative_error(S_test, S_lin_test, setting="mean_squared"))
        print("Nonlinear reconstruction training error = ", relative_error(S, S_approx, setting="mean_squared"))
        print("Nonlinear reconstruction test error = ", relative_error(S_test, S_approx_test, setting="mean_squared"))
    else:
        raise ValueError("Results have not been saved yet...Train the model first ! ")


def plot(config):
    tree_visualization(config)


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--mode', default="train", type=str, help='train or test')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    tol = float(config["params"]["tolerance"])
    alpha = config["add_params"]["alpha"]
    beta = config["add_params"]["beta"]
    p1 = config["params"]["p"]
    recompute_svd = config["add_params"]["compute_svd"]
    results_path = config["path_results"]
    approx_type = config["params"]["approximation_type"]
    setting = config["setting"]

    S, S_test, Sref, Sref_test = myloader(config)
    sref = Sref[:, 0]
    if approx_type == "sparse":
        method = CPN_S(S=S, Sref=Sref, tol_eps=tol, alpha=alpha, beta=beta, setting=setting)
    elif approx_type == "low_rank":
        method = CPN_LR(S=S, Sref=Sref, tol_eps=tol, alpha=alpha, beta=beta, setting=setting)
    else:
        raise ValueError("Approximation type not implemented !")

    if args.mode == "train":
        run()
    elif args.mode == "test":
        test()
    else:
        plot(config)
