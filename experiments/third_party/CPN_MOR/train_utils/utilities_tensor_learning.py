import tensap
from joblib import Parallel, delayed
import contextlib
import io
import sys

sys.path.append("train_utils")
from utilities import *

class CPN_LR(utils):

    def __init__(self, *args, **kwargs):
        super(CPN_LR, self).__init__(*args, **kwargs)

    @staticmethod
    def tensor_solver(SOLVER, input_data, output_data, ntrain):

        SOLVER.training_data = [None, output_data[:ntrain]]
        SOLVER.test_data = [input_data, output_data]

        st = io.StringIO()
        with contextlib.redirect_stdout(st), contextlib.redirect_stderr(st):
            F, _ = SOLVER.solve()
        return F

    def find_n(self, U, Q, p1, tol_min=1., n_min=False, train_set=1., Gamma=100):
        coeffs_dict = {}
        func = {}
        N_train = int(Q.shape[1] * train_set)
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
        lipschitz_consts = []
        lip_consts_inputs = []
        while indices_list:  # step
            learnt_w = []
            learnt_g = []
            lip_consts = []
            print(f"#################################################################step{dim}")

            X = [tensap.UniformRandomVariable(np.min(x), np.max(x)) for x in Q_check]
            BASIS = [
                tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(), range(p1 + 1))
                for x in X
            ]
            BASES = tensap.FunctionalBases(BASIS)
            SOLVER = tensap.TreeBasedTensorLearning.tensor_train_tucker(
                dim, tensap.SquareLossFunction()
            )

            SOLVER.bases = BASES
            SOLVER.bases_eval = BASES.eval(Q_check.T[:N_train, :])

            SOLVER.tolerance["on_stagnation"] = 1e-4

            SOLVER.initialization_type = "canonical"

            SOLVER.linear_model_learning.regularization = True
            # SOLVER.linear_model_learning.regularization_type = "l1"
            SOLVER.linear_model_learning.basis_adaptation = True
            SOLVER.linear_model_learning.error_estimation = True

            SOLVER.test_error = True
            # SOLVER.bases_eval_test = BASES.eval(X_TEST)

            SOLVER.rank_adaptation = True
            SOLVER.rank_adaptation_options["max_iterations"] = 20
            SOLVER.rank_adaptation_options["theta"] = 0.8
            SOLVER.rank_adaptation_options["early_stopping"] = True
            SOLVER.rank_adaptation_options["early_stopping_factor"] = 10

            if dim == 2:
                SOLVER.tree_adaptation = False
            else:
                SOLVER.tree_adaptation = True
            SOLVER.tree_adaptation_options["max_iterations"] = 1e2
            # SOLVER.tree_adaptation_options['force_rank_adaptation'] = True

            SOLVER.alternating_minimization_parameters["stagnation"] = 1e-5
            SOLVER.alternating_minimization_parameters["max_iterations"] = 50

            SOLVER.display = False
            SOLVER.alternating_minimization_parameters["display"] = False

            SOLVER.model_selection = True
            SOLVER.model_selection_options["type"] = "test_error"

            # SOLVER.loss_function.error_type = "absolute"

            coeffs = Q[indices_list, :].T
            SOLVER.tolerance["on_error"] = self.tol_eps * 0.1

            f = Parallel(n_jobs=-1)(
                delayed(self.tensor_solver)(SOLVER, Q_check.T, coeffs[:, i], N_train) for i in range(coeffs.shape[1]))

            pred = np.array([f[i](Q_check.T) for i in range(len(f))]).T

            for (i, j) in zip(indices_list, range(len(indices_list))):
                coeff_name = f'coef{i}'
                # g_i = self.gamma_i(i, indices_list, learnt_gamma_i, Gamma)
                g_i = self.gamma_i(i, indices_list, learnt_weights, Gamma)
                lip_const_i = self.lip_norm_i(Q_check, pred[:, j], index_r,
                                              np.array(lip_consts_inputs))

                if lp_error(coeffs[:, j], pred[:, j]) <= self.tol_eps_wise(i, indices_list,
                                                                           learnt_weights) and lip_const_i <= g_i:  # and lip_const_i <= g_i
                    func[coeff_name] = {'function': f[j], 'index': i, 'nb_deps': len(Q_check),
                                        'lip_constant': lip_const_i}
                    deleted_coeff = coeffs_dict.pop(coeff_name)
                    print("[", min(np.arange(dim)) + 1, "...", max(np.arange(dim)) + 1, "]", '|->',
                          deleted_coeff['index'] + 1)
                    w = self.weights(i, indices_list, learnt_weights)
                    learnt_w.append(w)
                    learnt_g.append(g_i)
                    lip_consts.append(lip_const_i)

            learnt_weights.extend(learnt_w)
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
                # gamma_list.append(self.gamma_i(func[next_coef]['index'], indices_list, learnt_gamma_i, Gamma))
                lip_consts_inputs.append(func[next_coef]['lip_constant'])
            lipschitz_consts.extend(lip_consts)
            # learnt_gamma_i.extend(learnt_g)
            indices_list = [value['index'] for value in coeffs_dict.values()]
            dim += 1

            print("Rest to learn = ", len(indices_list))

        print('Done !\t Dimension of the manifold = ', len(Qr))
        print('Index r = ', [r + 1 for r in index_r])

        return Qr, dict(sorted(func.items(), key=lambda x: x[1]["index"])), index_r, lipschitz_consts
