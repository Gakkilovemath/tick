from itertools import product

import numpy as np

from tick.inference.build.inference import (
    HawkesNonParamCumulant as _HawkesNonParamCumulant
)


class Cumulants(object):

    def __init__(self, half_width=100.):
        self.half_width = half_width
        self._cumulant = _HawkesNonParamCumulant(self.half_width)

    @property
    def realizations(self):
        return self._realizations

    @realizations.setter
    def realizations(self, val):
        self._realizations = val
        self.dim = len(self.realizations[0])
        self.n_realizations = len(self.realizations)
        self.time = np.zeros(self.n_realizations)
        for day, realization in enumerate(self.realizations):
            T_day = float(max(x[-1] for x in realization if len(x) > 0))
            self.time[day] = T_day
        self.L = np.zeros((self.n_realizations, self.dim))
        self.C = np.zeros((self.n_realizations, self.dim, self.dim))
        self._J = np.zeros((self.n_realizations, self.dim, self.dim))
        self._E_c = np.zeros((self.n_realizations, self.dim, self.dim, 2))
        self.K_c = np.zeros((self.n_realizations, self.dim, self.dim))

    def compute_cumulants(self):
        self.compute_L()
        self.compute_C_and_J()
        self.compute_E_c()
        self.K_c = [get_K_c(self._E_c[day]) for day in
                    range(self.n_realizations)]

    def compute_L(self):
        for day, realization in enumerate(self.realizations):
            L = np.zeros(self.dim)
            for i in range(self.dim):
                process = realization[i]
                L[i] = len(process) / self.time[day]
            self.L[day] = L.copy()

    def compute_C_and_J(self):
        d = self.dim

        for day in range(len(self.realizations)):
            C = np.zeros((d,d))
            J = np.zeros((d, d))
            for i, j in product(range(d), repeat=2):
                res = self._cumulant.compute_A_and_I_ij_rect(day, i, j, self.L[day][j])
                C[i, j] = res[0]
                J[i, j] = res[1]
            # we keep the symmetric part to remove edge effects
            C[:] = 0.5 * (C + C.T)
            J[:] = 0.5 * (J + J.T)
            self.C[day] = C.copy()
            self._J[day] = J.copy()

    def compute_E_c(self):
        d = self.dim

        for day in range(len(self.realizations)):
            E_c = np.zeros((d, d, 2))
            for i in range(d):
                for j in range(d):
                    E_c[i, j, 0] = self._cumulant.compute_E_ijk_rect(
                        day, i, j, j, self.L[day][i], self.L[day][j],
                        self._J[day][i, j])

                    E_c[i, j, 1] = self._cumulant.compute_E_ijk_rect(
                        day, j, j, i, self.L[day][j], self.L[day][j],
                        self._J[day][j, j])

            self._E_c[day] = E_c.copy()


###########
## Empirical cumulants with formula from the paper
###########

def get_K_c(E_c):
    K_c = np.zeros_like(E_c[:, :, 0])
    K_c += 2 * E_c[:, :, 0]
    K_c += E_c[:, :, 1]
    K_c /= 3.
    return K_c
