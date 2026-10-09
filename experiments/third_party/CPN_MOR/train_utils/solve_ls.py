import tensap

class solve_ls:
    def __init__(self, dim=1, I=None, maxIndex=1):
        self.p = maxIndex
        self.I = I
        self.d = dim

    def leastSquares(self, x, y, A, H, ls):
        ls.basis = None
        ls.basis_eval = A
        ls.training_data = [x, y]

        a, output = ls.solve()
        f = tensap.FunctionalBasisArray(a, H)
        ls.error_estimation = True
        return f
