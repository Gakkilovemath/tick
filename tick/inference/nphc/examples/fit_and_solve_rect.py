import matplotlib.pyplot as plt

import numpy as np
from scipy.linalg import inv

from tick.inference import HawkesExpKern
from tick.inference.hawkes_non_param_cumulant import NPHC
from tick.plot import plot_hawkes_kernel_norms
from tick.simulation import SimuHawkesExpKernels, SimuHawkesMulti

beta = 1.
mu = 0.01
d = 10
T = 1e6
H = 10
n_days = 20

baselines = mu * np.ones(d)
adjacency = np.zeros((d, d))
decays = np.ones((d, d))
for i in range(5):
    for j in range(5):
        if i <= j:
            adjacency[i][j] = 1.
            decays[i][j] = 10 * beta
for i in range(5, 10):
    for j in range(5, 10):
        if i >= j:
            adjacency[i][j] = 1.
            decays[i][j] = beta
adjacency /= 6

simu_hawkes = SimuHawkesExpKernels(baseline=baselines, adjacency=adjacency,
                                   decays=decays, end_time=T, verbose=False)
multi = SimuHawkesMulti(simu_hawkes, n_simulations=n_days, n_threads=-1)
multi.simulate()

nphc = NPHC(10, mu_true=baselines, R_true=inv(np.eye(d) - adjacency),
            alpha=.9, max_iter=300, print_every=20,
            step=1e-2, solver='adam')
nphc.fit(multi.timestamps)

R_pred = nphc.solve()
G_pred = np.eye(d) - inv(R_pred)

learner = HawkesExpKern(100)
learner.fit(multi.timestamps[0])

coeffs = np.hstack((baselines, G_pred.ravel()))
learner._set('coeffs', coeffs)
plot_hawkes_kernel_norms(learner, show=False)

plt.show()
