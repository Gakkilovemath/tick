from itertools import product

import numpy as np
from numba import jit

from tick.inference.build.inference import (
    HawkesNonParamCumulant as _HawkesNonParamCumulant
)


class Cumulants(object):

    def __init__(self, realizations, half_width=100.,
                 mu_true=None, R_true=None):
        self.realizations = realizations
        self.half_width = half_width
        self.sigma = self.half_width / 5.

        self.mu_true = mu_true
        self.R_true = R_true

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
        self.L_th = None
        self.C_th = None
        self.K_c_th = None
        self.R_true = None
        self.mu_true = None

        self._cumulant = _HawkesNonParamCumulant(self.half_width, self.sigma)
        self._cumulant.set_data(realizations, self.time)

    def compute_cumulants(self):
        self.compute_L()
        print("L is computed")

        self.compute_C_and_J()
        print("C is computed")

        self.compute_E_c()
        self.K_c = [get_K_c(self._E_c[day]) for day in
                    range(self.n_realizations)]
        print("K_c is computed")

        if self.R_true is not None and self.mu_true is not None:
            self.set_L_th()
            self.set_C_th()
            self.set_K_c_th()

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

    def set_R_true(self, R_true):
        self.R_true = R_true

    def set_mu_true(self, mu_true):
        self.mu_true = mu_true

    def set_L_th(self):
        assert self.R_true is not None, "You should provide R_true."
        assert self.mu_true is not None, "You should provide mu_true."
        self.L_th = get_L_th(self.mu_true, self.R_true)

    def set_C_th(self):
        assert self.R_true is not None, "You should provide R_true."
        self.C_th = get_C_th(self.L_th, self.R_true)

    def set_K_c_th(self):
        assert self.R_true is not None, "You should provide R_true."
        self.K_c_th = get_K_c_th(self.L_th, self.C_th, self.R_true)


###########
## Empirical cumulants with formula from the paper
###########

@jit
def get_K_c(E_c):
    K_c = np.zeros_like(E_c[:, :, 0])
    K_c += 2 * E_c[:, :, 0]
    K_c += E_c[:, :, 1]
    K_c /= 3.
    return K_c


##########
## Theoretical cumulants L, C, K, K_c
##########

@jit
def get_L_th(mu, R):
    return np.dot(R, mu)


@jit
def get_C_th(L, R):
    return np.dot(R, np.dot(np.diag(L), R.T))


@jit
def get_K_c_th(L, C, R):
    d = len(L)
    if R.shape[0] == d ** 2:
        R_ = R.reshape(d, d)
    else:
        R_ = R.copy()
    K_c = np.dot(C, (R_ * R_).T)
    K_c += 2 * np.dot(R_, (R_ * C).T)
    K_c -= 2 * np.dot(np.dot(R_, np.diag(L)), (R_ * R_).T)
    return K_c
