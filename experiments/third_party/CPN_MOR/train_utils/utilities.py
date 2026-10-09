import numpy as np

def lp_error(true, pred, setting="mean_squared"):
    if setting == "mean_squared":
        return np.linalg.norm(true - pred, ord=2)
    else:
        return max(abs(true - pred))


class utils(object):
    def __init__(self, S, Sref, tol_eps, alpha=1., beta=1 / np.sqrt(2), setting="mean_squared", **config):
        self.tol_eps = float(tol_eps)
        self.S = S
        self.Sref = Sref
        self.alpha = alpha
        self.beta = beta
        self.setting = setting
        self.config = config
        assert self.tol_eps < 1

    def truncate_svd(self, U):

        """
        Truncation of the left singular vectors matrix, to keep the N singular vectors needed to reach target precision.

        Args:
            U: Matrix containing the left singular vectors.

        Returns:
            - The truncated singular vectors matrix (basis of X_N)
            - The projected coordinates on the space X_N

        """
        print("Basis initial shape:  ", U.shape)
        print("Basis truncation...")
        ## Start at the end
        Q = U.T @ (self.S - self.Sref)
        r = Q.shape[0] - 1
        tol_check = (np.linalg.norm(self.S, 'fro') * (self.tol_eps * self.beta)) ** 2
        Q_norm = np.linalg.norm(Q[r, :], ord=2) ** 2
        while Q_norm <= tol_check:
            r -= 1
            Q_norm += np.linalg.norm(Q[r, :], ord=2) ** 2
        return U[:, :r + 1], Q[:r + 1, :]

    def greedy_snapshots(self, snapshot_matrix, tol_greedy=None, verbose=True):

        D, m = snapshot_matrix.shape

        self.e_infty_0 = max(np.linalg.norm(self.S, axis=0))
        if tol_greedy is None:
            tol_greedy = self.beta * self.tol_eps * self.e_infty_0

        V = np.zeros((D, 0))
        selected = []
        residuals = []

        current_approximation = np.zeros((D, m))

        for k in range(m):
            diffs = snapshot_matrix - current_approximation
            errors = np.linalg.norm(diffs, axis=0)

            for idx in selected:
                errors[idx] = -1.0

            max_err = np.max(errors)
            residuals.append(max_err)

            if verbose:
                print(f"Iter {k}: Max Error = {max_err:.3e}")

            if max_err < tol_greedy:
                break

            j_star = np.argmax(errors)
            selected.append(j_star)

            new_vec = diffs[:, j_star]
            new_vec /= np.linalg.norm(new_vec)

            new_vec_2d = new_vec.reshape(-1, 1)
            V = np.hstack((V, new_vec_2d))

            projections_on_new_basis = new_vec_2d.T @ snapshot_matrix
            current_approximation += new_vec_2d @ projections_on_new_basis

        Q = V.T @ (self.S - self.Sref)
        return V, Q  #, selected, residuals

    def greedy_snapshots_sphere(self, snapshot_matrix, tol_greedy=None, verbose=True):

        D, m = snapshot_matrix.shape
        S_centered = self.S - self.Sref

        self.e_infty_0 = max(np.linalg.norm(self.S, axis=0))
        if tol_greedy is None:
            tol_greedy = self.beta * self.tol_eps * self.e_infty_0

        selected = []

        mean_snapshot = np.mean(snapshot_matrix, axis=1)
        j_star = np.argmax(mean_snapshot @ snapshot_matrix)

        v_first = snapshot_matrix[:, j_star].reshape(-1, 1)

        V = v_first.reshape(-1, 1)
        selected.append(j_star)

        residuals = [1.]

        current_approximation = v_first @ (v_first.T @ snapshot_matrix)
        current_approximation_snapshots = v_first @ (v_first.T @ S_centered)

        for k in range(1, m):

            diffs = snapshot_matrix - current_approximation
            diffs_snapshots = S_centered - current_approximation_snapshots

            errors = np.linalg.norm(diffs, axis=0)
            errors_snapshots = np.linalg.norm(diffs_snapshots, axis=0)

            for idx in selected:
                errors[idx] = -1.0

            max_err = np.max(errors)
            max_err_snapshots = np.max(errors_snapshots)
            residuals.append(max_err)

            if verbose:
                print(f"Iter {k}: Max Error = {max_err:.3e}")

            if max_err_snapshots < tol_greedy:
                break

            j_star = np.argmax(errors)
            selected.append(j_star)

            new_vec = diffs[:, j_star]
            new_vec /= np.linalg.norm(new_vec)

            new_vec_2d = new_vec.reshape(-1, 1)
            V = np.hstack((V, new_vec_2d))

            projections_on_new_basis = new_vec_2d.T @ snapshot_matrix
            current_approximation += new_vec_2d @ projections_on_new_basis

            projections_on_new_basis_snapshots = new_vec_2d.T @ S_centered
            current_approximation_snapshots += new_vec_2d @ projections_on_new_basis_snapshots

        Q = V.T @ S_centered
        return V, Q  #, selected, residuals

    def greedy_snapshots_sphere_2(self, snapshot_matrix, tol_greedy=None, verbose=True):

        D, m = snapshot_matrix.shape
        S_centered = self.S - self.Sref

        self.e_infty_0 = max(np.linalg.norm(self.S, axis=0))
        if tol_greedy is None:
            tol_greedy = self.beta * self.tol_eps * self.e_infty_0

        V = np.zeros((D, 0))
        selected = []
        residuals = []

        current_approximation = np.zeros((D, m))
        current_approximation_snapshots = np.zeros(S_centered.shape)

        for k in range(m):
            diffs = snapshot_matrix - current_approximation
            errors = np.linalg.norm(diffs, axis=0)

            diffs_snapshots = S_centered - current_approximation_snapshots
            errors_snapshots = np.linalg.norm(diffs_snapshots, axis=0)

            for idx in selected:
                errors[idx] = -1.0

            max_err = np.max(errors)
            residuals.append(max_err)

            max_err_snapshots = np.max(errors_snapshots)

            if verbose:
                print(f"Iter {k}: Max Error = {max_err:.3e}")

            if max_err_snapshots < tol_greedy:
                break

            j_star = np.argmax(errors)
            selected.append(j_star)

            new_vec = diffs[:, j_star]
            new_vec /= np.linalg.norm(new_vec)

            new_vec_2d = new_vec.reshape(-1, 1)
            V = np.hstack((V, new_vec_2d))

            projections_on_new_basis = new_vec_2d.T @ snapshot_matrix
            current_approximation += new_vec_2d @ projections_on_new_basis

            projections_on_new_basis_snapshots = new_vec_2d.T @ S_centered
            current_approximation_snapshots += new_vec_2d @ projections_on_new_basis_snapshots

        Q = V.T @ (self.S - self.Sref)
        return V, Q  #, selected, residuals

    def weights(self, index, indices_list, learnt_weights):
        X = 1 - np.sum(learnt_weights)
        w = (index ** (self.alpha) / np.sum(np.array(indices_list) ** (self.alpha))) * X
        return w

    def new_weights(self, abs_error):
        return abs_error ** 2 / ((1 - self.beta ** 2) * (self.tol_eps * np.linalg.norm(self.S, 'fro')) ** 2)

    @staticmethod
    def new_weights_gamma(lip_const_i, Gamma):
        return lip_const_i ** 2 / (Gamma ** 2 - 1)

    def tol_eps_wise(self, index, indices_list, learnt_weights):

        if self.setting == "mean_squared":
            return (self.tol_eps * np.sqrt(self.weights(index, indices_list,
                                                        learnt_weights)) * np.linalg.norm(self.S, 'fro')) * np.sqrt(
                1 - self.beta ** 2)
        else:
            return self.weights(index, indices_list, learnt_weights) * (1 - self.beta) * self.tol_eps * self.e_infty_0

    @staticmethod
    def norm_i(b, index_n, gamma_list):

        if len(b.shape) == 1:
            return np.max(np.linalg.norm(b[index_n], ord=2), (1 / gamma_list) * np.abs(np.delete(b, index_n, axis=0)))
        elif len(b.shape) == 2:
            if len(gamma_list) == 0:
                return np.linalg.norm(b[index_n], ord=2, axis=0)
            else:
                return np.maximum(np.linalg.norm(b[index_n], ord=2, axis=0),
                                  np.max((1 / gamma_list)[:, np.newaxis] * np.abs(np.delete(b, index_n, axis=0)),
                                         axis=0))
        elif len(b.shape) == 3:
            if len(gamma_list) == 0:
                return np.linalg.norm(b[index_n], ord=2, axis=0)
            else:
                return np.maximum(np.linalg.norm(b[index_n], ord=2, axis=0), np.max(
                    (1 / gamma_list)[:, np.newaxis, np.newaxis] * np.abs(np.delete(b, index_n, axis=0)), axis=0))

    def lip_norm_i(self, X, Y, index_n, gamma_list):

        diff_1 = X[index_n, :, np.newaxis] - X[index_n, np.newaxis, :]
        diff_2 = np.delete(X[:, :, np.newaxis], index_n, axis=0) - np.delete(X[:, np.newaxis, :], index_n, axis=0)
        diff = np.concatenate((diff_1, diff_2), axis=0)
        norm_diff = self.norm_i(diff, index_n, gamma_list)
        norm_diff = np.where(norm_diff == 0, np.inf, norm_diff)
        closest_neighbor = np.argmin(norm_diff, axis=1)
        X_close = X[:, closest_neighbor]
        Y_close = Y[closest_neighbor]

        ratio = np.abs(Y - Y_close) / self.norm_i(X - X_close, index_n, gamma_list)
        return np.max(ratio)

    @staticmethod
    def lip_norm_2(X, Y):

        diff = X[:, :, np.newaxis] - X[:, np.newaxis, :]
        norm_diff = np.linalg.norm(diff, ord=2, axis=0)
        norm_diff = np.where(norm_diff == 0, np.inf, norm_diff)
        closest_neighbor = np.argmin(norm_diff, axis=1)
        X_close = X[:, closest_neighbor]
        Y_close = Y[closest_neighbor]

        ratio = np.abs(Y - Y_close) / np.linalg.norm(X - X_close, axis=0)
        return np.max(ratio)

    def gamma_i(self, index, indices_list, learnt_weights, Gamma):
        gamma = np.sqrt(self.weights(index, indices_list, learnt_weights) * (Gamma ** 2 - 1))
        return gamma

    def new_gamma_i(self, new_weights, Gamma):
        gamma = np.sqrt(new_weights * (Gamma ** 2 - 1))
        return gamma

    def find_n_min(self, U, Q, tol_min):
        n = 1
        S_approx = self.Sref + U[:, :n] @ Q[:n, :]
        err = tol_min * np.linalg.norm(self.S, 'fro')
        while np.linalg.norm(self.S - S_approx, 'fro') > err:
            n += 1
            S_approx = self.Sref + U[:, :n] @ Q[:n, :]
        return n
