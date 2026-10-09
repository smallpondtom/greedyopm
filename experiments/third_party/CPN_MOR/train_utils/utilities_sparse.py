from joblib import Parallel, delayed
import sys

sys.path.append("train_utils")
from solve_ls import *
from utilities import *
import contextlib
import io


class CPN_S(utils):
    def __init__(self, *args, **kwargs):
        super(CPN_S, self).__init__(*args, **kwargs)

    @staticmethod
    def sparse_solver(s, input_data, output_data, A, H, ls):

        st = io.StringIO()
        with contextlib.redirect_stdout(st), contextlib.redirect_stderr(st):
            f = s.leastSquares(input_data, output_data, A, H, ls)

        return f

    def find_n(self, U, Q, p1, tol_min=False, n_min=False, train_set=1., Gamma=100):
        coeffs_dict = {}
        func = {}
        N_train = int(Q.shape[1] * train_set)
        N_val = Q.shape[1] - N_train
        for j, q_j in enumerate(Q):
            coeff_name = f"coef{j}"
            coeffs_dict[coeff_name] = {'value': q_j, 'index': j}

        if not n_min:
            n = self.find_n_min(U, Q, float(tol_min))
        else:
            n = n_min
        print("n min     =   ", n)
        index_r = [r for r in range(n)]
        for k in index_r:
            coeffs_dict.pop(f"coef{k}")
        indices_list = [value["index"] for value in coeffs_dict.values()]
        dim = n

        Qr = Q[:n, :]
        Q_check = Q[:n, :]

        learnt_weights = []
        learnt_weights_gamma = []
        lipschitz_consts = []
        lip_consts_inputs = []

        ls = tensap.LinearModelLearningSquareLoss()
        ls.regularization = True
        ls.regularization_type = "l1"
        ls.regularization_options = {"alpha": 0.}
        ls.model_selection = True
        if ls.regularization_type == "l2":
            ls.model_selection = False
        ls.error_estimation = True
        ls.error_estimation_type = 'leave_out'
        ls.error_estimation_options["correction"] = True
        while indices_list:  # step
            learnt_w = []
            learnt_w_gamma = []
            lip_consts = []
            print(f"#################################################################step{dim}")
            X = [tensap.UniformRandomVariable(np.min(x), np.max(x)) for x in Q_check]
            BASIS = [
                tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(), range(p1 + 1))
                for x in X
            ]
            BASES = tensap.FunctionalBases(BASIS)
            I = tensap.MultiIndices.hyperbolic_cross_set(dim, p1)
            # I = tensap.MultiIndices.with_bounded_norm(dim, 1, p1)
            s = solve_ls(dim=dim, I=I, maxIndex=p1)
            H = tensap.SparseTensorProductFunctionalBasis(BASES, I)
            A = H.eval(Q_check.T[:N_train, :])
            coeffs = Q[indices_list, :].T
            f = Parallel(n_jobs=-1)(
                delayed(self.sparse_solver)(s, Q_check.T[:N_train, :], coeffs[:N_train, i], A, H, ls)
                for i in range(coeffs.shape[1]))
            pred = np.array([f[i](Q_check.T) for i in range(len(f))]).T
            for (i, j) in zip(indices_list, range(len(indices_list))):
                coeff_name = f'coef{i}'
                g_i = self.gamma_i(i, indices_list, learnt_weights_gamma, Gamma)
                lip_const_i = self.lip_norm_i(Q_check, pred[:, j], index_r,
                                              np.array(lip_consts_inputs))

                if lp_error(coeffs[-N_val:, j], pred[-N_val:, j]) <= self.tol_eps_wise(i, indices_list,
                                                                                       learnt_weights):# and lip_const_i <= g_i:
                    func[coeff_name] = {'function': f[j], 'index': i, 'nb_deps': len(Q_check),
                                        'lip_constant': lip_const_i}
                    deleted_coeff = coeffs_dict.pop(coeff_name)
                    print("[", min(np.arange(dim)) + 1, "...", max(np.arange(dim)) + 1, "]", '|->',
                          deleted_coeff['index'] + 1)
                    w_i = self.new_weights(lp_error(coeffs[-N_val:, j], pred[-N_val:, j]))
                    w_i_gamma = self.new_weights_gamma(lip_const_i, Gamma)
                    learnt_w.append(w_i)
                    learnt_w_gamma.append(w_i_gamma)
                    lip_consts.append(lip_const_i)

            learnt_weights.extend(learnt_w)
            learnt_weights_gamma.extend(learnt_w_gamma)
            next_coef = f'coef{dim}'
            if next_coef in coeffs_dict:
                q_next = coeffs_dict.pop(next_coef)
                Qr = np.concatenate([Qr, q_next['value'][None,]], axis=0)
                index_r.append(q_next['index'])
                Q_check = np.concatenate([Q_check, q_next['value'][None,]], axis=0)
            else:
                nb_deps = func[next_coef]['nb_deps']
                q_next_approx = func[next_coef]['function'](Q_check.T[:, :nb_deps])
                Q_check = np.concatenate([Q_check, q_next_approx[None,]], axis=0)
                lip_consts_inputs.append(func[next_coef]['lip_constant'])
            lipschitz_consts.extend(lip_consts)
            indices_list = [value['index'] for value in coeffs_dict.values()]
            dim += 1

            print("Rest to learn = ", len(indices_list))

        print('Done !\t Dimension of the manifold = ', len(Qr))
        print('I = ', [r + 1 for r in index_r])

        return Qr, dict(sorted(func.items(), key=lambda x: x[1]["index"])), index_r, lipschitz_consts
