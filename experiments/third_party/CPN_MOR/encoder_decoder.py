from dataclasses import dataclass
import numpy as np

@dataclass
class Encoder:
    index: np.ndarray
    basis: np.ndarray
    u_ref: np.ndarray

    def __call__(self, u):
        u_ref = self.u_ref
        if len(u.shape) == 2:
            u_ref = np.tile(self.u_ref, (u.shape[1], 1)).T
        return self.basis.T @ (u - u_ref)

@dataclass
class Decoder:
    linear_index: np.ndarray
    nonlinear_index: np.ndarray
    linear_basis: np.ndarray
    nonlinear_basis: np.ndarray
    functions_f: dict
    # eval_all_coeffs: callable
    u_ref: np.ndarray

    def eval_all_coeffs(self, Q):
        Qbar = np.zeros((len(self.functions_f.items()), Q.shape[1]))
        list_keys = list(self.functions_f.keys())
        Q_total = np.zeros((len(self.linear_index) + len(self.functions_f.items()), Q.shape[1]))
        Q_total[self.linear_index, :] = Q
        for i, coef_name in enumerate(list_keys):
            nb_deps = self.functions_f[coef_name]['nb_deps']
            Qbar[i, :] = self.functions_f[coef_name]['function'](Q_total.T[:, :nb_deps])
            Q_total[self.functions_f[coef_name]['index']] = Qbar[i, :]
        return Qbar

    def __call__(self, a):
        if len(a.shape) == 1:
            a = np.expand_dims(a, 1)
        u_ref = np.array([self.u_ref] * a.shape[1]).T
        a_bar = self.eval_all_coeffs(a)
        u_pred = u_ref + self.linear_basis @ a + self.nonlinear_basis @ a_bar
        if u_pred.shape[1] == 1:
            u_pred = np.squeeze(u_pred)
        return u_pred

    def functions_g(self, index: int, Qr):
        if len(Qr.shape) == 1:
            Qr = np.expand_dims(Qr, 1)
        if index in self.linear_index:
            return Qr
        elif index in self.nonlinear_index:
            coef_name = f"coef{index}"
            nb_deps_coef = self.functions_f[coef_name]['nb_deps']
            Q_total = np.zeros((nb_deps_coef, Qr.shape[1]))
            Q_total[self.linear_index, :] = Qr
            for k in np.delete(np.arange(nb_deps_coef), self.linear_index):
                inter_coef = f"coef{k}"
                nb_deps = self.functions_f[inter_coef]['nb_deps']
                Q_total[self.functions_f[inter_coef]['index'], :] = self.functions_f[inter_coef]['function'](Q_total.T[:, :nb_deps])

            return self.functions_f[coef_name]['function'](Q_total.T)
        else:
            raise ValueError("Coefficient does not exist")
