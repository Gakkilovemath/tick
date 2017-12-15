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
        h_w = self.half_width
        d = self.dim

        E_ijk = E_ijk_rect

        for day in range(len(self.realizations)):
            realization = self.realizations[day]
            E_c = np.zeros((d, d, 2))
            for i in range(d):
                for j in range(d):
                    E_c[i, j, 0] = E_ijk(realization[i], realization[j],
                                         realization[j], -h_w, h_w,
                                         self.time[day], self.L[day][i],
                                         self.L[day][j], self._J[day][i, j],
                                         self.sigma)
                    E_c[i, j, 1] = E_ijk(realization[j], realization[j],
                                         realization[i], -h_w, h_w,
                                         self.time[day], self.L[day][j],
                                         self.L[day][j], self._J[day][j, j],
                                         self.sigma)
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


@jit
def E_ijk_rect(realization_i, realization_j, realization_k, a, b, T, L_i, L_j, J_ij, sigma=1.0):
    """
    Computes the mean of the centered product of i's and j's jumps between \tau + a and \tau + b, that is
    \frac{1}{T} \sum_{\tau \in Z^k} ( N^i_{\tau + b} - N^i_{\tau + a} - \Lambda^i * ( b - a ) )
                                  * ( N^j_{\tau + b} - N^j_{\tau + a} - \Lambda^j * ( b - a ) )
    """
    res = 0
    u = 0
    x = 0
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]
    n_k = realization_k.shape[0]

    trend_i = L_i * (b - a)
    trend_j = L_j * (b - a)

    for t in range(n_k):
        tau = realization_k[t]

        if tau + a < 0: continue

        # work on realization_i
        while u < n_i:
            if realization_i[u] <= tau + a:
                u += 1
            else:
                break
        v = u

        while v < n_i:
            if realization_i[v] < tau + b:
                v += 1
            else:
                break

        # work on realization_j
        while x < n_j:
            if realization_j[x] <= tau + a:
                x += 1
            else:
                break
        y = x

        while y < n_j:
            if realization_j[y] < tau + b:
                y += 1
            else:
                break
        if y == n_j or v == n_i: continue

        res += (v - u - trend_i) * (y - x - trend_j) - J_ij
    res /= T
    return res

def worker_day_C_J(fun, realization, h_w, T, L, sigma, d):
    C = np.zeros((d, d))
    J = np.zeros((d, d))
    for i, j in product(range(d), repeat=2):
        if len(realization[i])*len(realization[j]) != 0:
            z = fun(realization[i], realization[j], h_w, T, L[j], sigma)
            C[i,j] = z.real
            J[i,j] = z.imag
    return C + J * 1j

def worker_day_E(fun, realization, h_w, T, L, J, sigma, d):
    E_c = np.zeros((d, d, 2))
    for i, j in product(range(d), repeat=2):
        if len(realization[i])*len(realization[j]) != 0:
            E_c[i, j, 0] = fun(realization[i], realization[j], realization[j], -h_w, h_w,
                                  T, L[i], L[j], J[i, j], sigma)
            E_c[i, j, 1] = fun(realization[j], realization[j], realization[i], -h_w, h_w,
                                  T, L[j], L[j], J[j, j], sigma)
    return E_c
