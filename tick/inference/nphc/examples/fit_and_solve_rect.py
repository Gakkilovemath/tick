import matplotlib.pyplot as plt

import numpy as np
from scipy.linalg import inv

import tick.simulation as hk
from tick.inference.hawkes_non_param_cumulant import NPHC

#####################################################
### Simulation of a 10-dimensional Hawkes process ###
#####################################################
beta = 1.
mu = 0.01
d = 10
T = 1e6
H = 10
n_days = 20 

mus = mu * np.ones(d)
Alpha = np.zeros((d,d))
Beta = np.zeros((d,d))
for i in range(5):
    for j in range(5):
        if i <= j:
            Alpha[i][j] = 1.
            Beta[i][j] = 100*beta
for i in range(5,10):
    for j in range(5,10):
        if i >= j:
            Alpha[i][j] = 1.
            Beta[i][j] = beta
Alpha /= 6

ticks = []
kernels = [[hk.HawkesKernelExp(a, b) for (a, b) in zip(a_list, b_list)] for (a_list, b_list) in zip(Alpha, Beta)]
for _ in range(n_days):
    h = hk.SimuHawkes(kernels=kernels, baseline=list(mus), end_time=T)
    h.simulate()
    ticks.append(h.timestamps)


######################################
### Fit (=> compute the cumulants) ###
######################################
nphc = NPHC()
nphc.fit(ticks, half_width=10, mu_true=mus, R_true=inv(np.eye(d) - Alpha))
# print mean error of cumulants estimation

#################################################
### Solve (=> minimize the objective function ###
#################################################
R_pred = nphc.solve(alpha=.9,training_epochs=300,display_step=20,learning_rate=1e-2,optimizer='adam')

# print final error of estimation
G_pred = np.eye(d) - inv(R_pred)

print(Alpha)
print(G_pred)

from tick.plot import plot_hawkes_kernel_norms
from tick.inference import HawkesExpKern
learner = HawkesExpKern(100)
learner.fit(ticks[0])

coeffs = np.hstack((mus, G_pred.ravel()))
learner._set('coeffs', coeffs)
plot_hawkes_kernel_norms(learner, show=True)
