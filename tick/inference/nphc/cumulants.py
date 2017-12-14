from itertools import product

import numpy as np
from joblib import Parallel, delayed
from numba import jit
from math import erf
from numpy import sqrt, pi, exp
from scipy.stats import norm


class Cumulants(object):

    def __init__(self, realizations, half_width=100., filtr='rectangular',
                 method="parallel", mu_true=None, R_true=None):
        self.realizations = realizations
        self.half_width = half_width
        self.sigma = self.half_width / 5.

        if filtr not in ['rectangular', 'gaussian']:
            raise ValueError("`filtr` should either equal `rectangular` "
                             "or `gaussian`, recieved {}".format(filtr))
        self.filtr = filtr

        if method not in ['classic', 'parallel_by_day',
                          'parallel_by_component']:
            raise ValueError("`method` should either equal `parallel_by_day`, "
                             "`parallel_by_component` or `classic`, "
                             "recieved {}".format(method))

        self.method = method
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
        h_w = self.half_width
        d = self.dim

        if self.filtr == "rectangular":
            A_and_I_ij = A_and_I_ij_rect
        elif self.filtr == "gaussian":
            A_and_I_ij = A_and_I_ij_gauss

        if self.method == 'classic':
            for day in range(len(self.realizations)):
                realization = self.realizations[day]
                C = np.zeros((d,d))
                J = np.zeros((d, d))
                for i, j in product(range(d), repeat=2):
                    z = A_and_I_ij(realization[i], realization[j], h_w, self.time[day], self.L[day][j], self.sigma)
                    C[i,j] = z.real
                    J[i,j] = z.imag
                # we keep the symmetric part to remove edge effects
                C[:] = 0.5 * (C + C.T)
                J[:] = 0.5 * (J + J.T)
                self.C[day] = C.copy()
                self._J[day] = J.copy()

        elif self.method == 'parallel_by_day':
            l = Parallel(-1)(delayed(worker_day_C_J)(A_and_I_ij, realization, h_w, T, L, self.sigma, d) for (realization, T, L) in zip(self.realizations, self.time, self.L))
            self.C = [0.5*(z.real+z.real.T) for z in l]
            self._J = [0.5*(z.imag+z.imag.T) for z in l]

        elif self.method == 'parallel_by_component':
            for day in range(len(self.realizations)):
                realization = self.realizations[day]
                l = Parallel(-1)(
                        delayed(A_and_I_ij)(realization[i], realization[j], h_w, self.time[day], self.L[day][j], self.sigma)
                        for i in range(d) for j in range(d))
                C_and_J = np.array(l).reshape(d, d)
                C = C_and_J.real
                J = C_and_J.imag
                # we keep the symmetric part to remove edge effects
                C[:] = 0.5 * (C + C.T)
                J[:] = 0.5 * (J + J.T)
                self.C[day] = C.copy()
                self._J[day] = J.copy()

    def compute_E_c(self):
        h_w = self.half_width
        d = self.dim

        if self.filtr == "rectangular":
            E_ijk = E_ijk_rect
        elif self.filtr == "gaussian":
            E_ijk = E_ijk_gauss

        if self.method == 'classic':
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

        elif self.method == 'parallel_by_day':
            self._E_c = Parallel(-1)(delayed(worker_day_E)(E_ijk, realization, h_w, T, L, J, self.sigma, d) for (realization, T, L, J) in zip(self.realizations, self.time, self.L, self._J))

        elif self.method == 'parallel_by_component':
            for day in range(len(self.realizations)):
                realization = self.realizations[day]
                E_c = np.zeros((d, d, 2))
                l1 = Parallel(-1)(
                        delayed(E_ijk)(realization[i], realization[j], realization[j], -h_w, h_w,
                                            self.time[day], self.L[day][i], self.L[day][j], self._J[day][i, j], self.sigma) for i in range(d) for j in range(d))
                l2 = Parallel(-1)(
                        delayed(E_ijk)(realization[j], realization[j], realization[i], -h_w, h_w,
                                            self.time[day], self.L[day][j], self.L[day][j], self._J[day][j, j], self.sigma) for i in range(d) for j in range(d))
                E_c[:, :, 0] = np.array(l1).reshape(d, d)
                E_c[:, :, 1] = np.array(l2).reshape(d, d)
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


##########
## Useful fonctions to set_ empirical integrated cumulants
##########

#@jit
#def filtr_fun(X, sigma, filtr='rectangular'):
#    if filtr == 'rectangular':
#        return np.ones_like(X)
#    elif filtr == 'gaussian':
#        return sigma * sqrt(2 * pi) * norm.pdf(X, scale=sigma)
#    else:
#        return np.zeros_like(X)


# @jit(double(double[:],double[:],int32,int32,double,double,double), nogil=True, nopython=True)
# @jit(float64(float64[:],float64[:],int64,int64,int64,float64,float64), nogil=True, nopython=True)
@jit
def A_ij_rect(realization_i, realization_j, a, b, T, L_j):
    """
    Computes the mean centered number of jumps of N^j between \tau + a and \tau + b, that is
    \frac{1}{T} \sum_{\tau \in Z^i} ( N^j_{\tau + b} - N^j_{\tau + a} - \Lambda^j (b - a) )
    """
    res = 0
    u = 0
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]

    trend_j = L_j * (b - a)

    for t in range(n_i):
        # count the number of jumps
        tau = realization_i[t]
        if tau + a < 0: continue
        while u < n_j:
            if realization_j[u] <= tau + a:
                u += 1
            else:
                break

        v = u
        while v < n_j:
            if realization_j[v] < tau + b:
                v += 1
            else:
                break
        if v == n_j: continue
        res += v - u - trend_j
    res /= T
    return res


@jit
def A_ij_gauss(realization_i, realization_j, a, b, T, L_j, sigma=1.0):
    """
    Computes the mean centered number of jumps of N^j between \tau + a and \tau + b, that is
    \frac{1}{T} \sum_{\tau \in Z^i} ( N^j_{\tau + b} - N^j_{\tau + a} - \Lambda^j (b - a) )
    """
    res = 0
    u = 0
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]

    trend_j = L_j * sigma * sqrt(2 * pi) * (norm.cdf(b/sigma) - norm.cdf(a/sigma))

    for t in range(n_i):
        # count the number of jumps
        tau = realization_i[t]
        if tau + a < 0: continue
        while u < n_j:
            if realization_j[u] <= tau + a:
                u += 1
            else:
                break
        v = u
        sub_res = 0.
        while v < n_j:
            if realization_j[v] < tau + b:
                sub_res += exp(-.5*((realization_j[v]-tau)/sigma)**2)
                v += 1
            else:
                break
        if v == n_j: continue
        res += sub_res - trend_j
    res /= T
    return res

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


@jit(nopython=True)
def E_ijk_gauss(realization_i, realization_j, realization_k, a, b, T, L_i, L_j, J_ij, sigma=1.0):
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

    trend_i = L_i * sigma * sqrt(2 * pi) * (norm_cdf(b/sigma) - norm_cdf(a/sigma))
    trend_j = L_j * sigma * sqrt(2 * pi) * (norm_cdf(b/sigma) - norm_cdf(a/sigma))

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
        sub_res_i = 0.
        while v < n_i:
            if realization_i[v] < tau + b:
                sub_res_i += exp(-.5*((realization_i[v]-tau)/sigma)**2)
                v += 1
            else:
                break
        if v == n_i: continue

        # work on realization_j
        while x < n_j:
            if realization_j[x] <= tau + a:
                x += 1
            else:
                break
        y = x
        sub_res_j = 0.
        while y < n_j:
            if realization_j[y] < tau + b:
                sub_res_j += exp(-.5*((realization_j[y]-tau)/sigma)**2)
                y += 1
            else:
                break
        if y == n_j: continue
        res += (sub_res_i - trend_i) * (sub_res_j - trend_j) - J_ij
    res /= T
    return res

@jit
def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

@jit
def A_and_I_ij_rect(realization_i, realization_j, half_width, T, L_j, sigma=1.0):
    """
    Computes the integral \int_{(0,H)} t c^{ij} (t) dt. This integral equals
    \frac{1}{T} \sum_{\tau \in Z^i} \sum_{\tau' \in Z^j} [ (\tau - \tau') 1_{ \tau - H < \tau' < \tau } - H^2 / 2 \Lambda^j ]
    """
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]
    res_C = 0
    res_J = 0
    u = 0
    width = 2 * half_width
    trend_C_j = L_j * width
    trend_J_j = L_j * width ** 2

    for t in range(n_i):
        tau = realization_i[t]
        tau_minus_half_width = tau - half_width
        tau_minus_width = tau - width

        if tau_minus_half_width < 0: continue

        while u < n_j:
            if realization_j[u] <= tau_minus_width:
                u += 1
            else:
                break
        v = u
        w = u
        sub_res = 0.
        while v < n_j:
            tau_p_minus_tau = realization_j[v] - tau
            if tau_p_minus_tau < -half_width:
                sub_res += width + tau_p_minus_tau
                v += 1
            elif tau_p_minus_tau < 0:
                sub_res += width + tau_p_minus_tau
                w += 1
                v += 1
            elif tau_p_minus_tau < half_width:
                sub_res += width - tau_p_minus_tau
                w += 1
                v += 1
            elif tau_p_minus_tau < width:
                sub_res += width - tau_p_minus_tau
                v += 1
            else:
                break
        if v == n_j: continue
        res_C += w - u - trend_C_j
        res_J += sub_res - trend_J_j
    res_C /= T
    res_J /= T
    return res_C + res_J * 1j


@jit
def A_and_I_ij_gauss(realization_i, realization_j, half_width, T, L_j, sigma=1.0):
    """
    Computes the integral \int_{(0,H)} t c^{ij} (t) dt. This integral equals
    \frac{1}{T} \sum_{\tau \in Z^i} \sum_{\tau' \in Z^j} [ (\tau - \tau') 1_{ \tau - H < \tau' < \tau } - H^2 / 2 \Lambda^j ]
    """
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]
    res_C = 0
    res_J = 0
    u = 0
    width = sqrt(2) * half_width
    trend_C_j = L_j * sigma * sqrt(2 * pi) * (norm_cdf(half_width/sigma) - norm_cdf(-half_width/sigma))
    trend_J_j = L_j * sigma**2 * 2 * pi * (norm_cdf(half_width/(sqrt(2)*sigma)) - norm_cdf(-half_width/(sqrt(2)*sigma)))

    for t in range(n_i):
        tau = realization_i[t]
        tau_minus_half_width = tau - half_width
        tau_minus_width = tau - width

        if tau_minus_half_width < 0: continue

        while u < n_j:
            if realization_j[u] <= tau_minus_width:
                u += 1
            else:
                break
        v = u
        w = u
        sub_res_C = 0.
        sub_res_J = 0.
        while v < n_j:
            tau_p_minus_tau = realization_j[v] - tau
            if tau_p_minus_tau < -half_width:
                sub_res_J += sigma*sqrt(pi)*exp(-.25*(tau_p_minus_tau/sigma)**2)
                v += 1
            elif tau_p_minus_tau < half_width:
                sub_res_C += exp(-.5*(tau_p_minus_tau/sigma)**2)
                sub_res_J += sigma*sqrt(pi)*exp(-.25*(tau_p_minus_tau/sigma)**2)
                v += 1
            elif tau_p_minus_tau < width:
                sub_res_J += sigma*sqrt(pi)*exp(-.25*(tau_p_minus_tau/sigma)**2)
                v += 1
            else:
                break
        if v == n_j: continue
        res_C += sub_res_C - trend_C_j
        res_J += sub_res_J - trend_J_j
    res_C /= T
    res_J /= T
    return res_C + res_J * 1j


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
