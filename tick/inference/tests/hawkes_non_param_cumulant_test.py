# License: BSD 3 clause

import unittest

import numpy as np

from tick.inference import NPHC
from tick.inference.tests.inference import InferenceTest
from tick.simulation import SimuHawkesExpKernels, SimuHawkesMulti


class Test(InferenceTest):
    def setUp(self):
        self.dim = 2
        np.random.seed(320982)

    @staticmethod
    def get_train_data(n_nodes=3, decay=1.):
        np.random.seed(130947)
        baseline = np.random.rand(n_nodes) + 0.1
        adjacency = np.random.rand(n_nodes, n_nodes)
        if isinstance(decay, (int, float)):
            decay = np.ones((n_nodes, n_nodes)) * decay

        sim = SimuHawkesExpKernels(adjacency=adjacency, decays=decay,
                                   baseline=baseline, verbose=False,
                                   seed=13487, end_time=1000)
        sim.adjust_spectral_radius(0.8)
        adjacency = sim.adjacency
        multi = SimuHawkesMulti(sim, n_simulations=3)

        multi.simulate()
        return multi.timestamps, baseline, adjacency

    def test_hawkes_nphc_cumulants(self):
        timestamps, baseline, adjacency = Test.get_train_data(decay=3.)

        expected_L = [[2.18034168, 2.8594481, 4.45869871],
                      [1.99015833, 2.6472106, 4.16933169],
                      [2.27845603, 2.89257894, 4.7639535]]

        expected_C = [[[13.32657839, 10.11904124, 14.46051284],
                       [10.11904124, 10.83865994, 7.95432054],
                       [14.46051284, 7.95432054, 4.04515646]],

                      [[31.96293101, 41.98883905, 64.8096231],
                       [41.98883905, 56.10860297, 87.4726287],
                       [64.8096231, 87.4726287, 136.10384119]],

                      [[1.76797049, -1.16693364, 11.42660805],
                       [-1.16693364, 4.34865034, 14.36453405],
                       [11.42660805, 14.36453405, 58.66427053]]]

        expected_K = [
            np.array([[3223.9240599, 2971.38679862, 20341.58074785],
                      [3117.08347713, 2766.78958765, 24361.53181028],
                      [7910.20155272, 9768.15699752, 52986.95555252]]),
            np.array([[-5486.61240309, -9898.42199218, -23297.49526655],
                      [-7383.95043795, -13230.37717013, -31444.01986259],
                      [-11335.48717443, -20501.72973439, -47614.84883139]]),
            np.array([[2410.2256186, 4049.2962653, 1265.32736146],
                      [3205.74610429, 4797.78697804, 1563.66300526],
                      [2798.54371545, 4421.71603128, -5824.92071766]])]

        model = NPHC(100.)
        model.fit(timestamps)
        model._compute_cumulants()

        np.testing.assert_array_almost_equal(model.L, expected_L)
        np.testing.assert_array_almost_equal(model.C, expected_C)
        np.testing.assert_array_almost_equal(model.K_c, expected_K)

        self.assertAlmostEqual(model.approximate_optimal_alpha(),
                               0.999197628503)

    def test_hawkes_nphc_cumulants_solve(self):
        timestamps, baseline, adjacency = Test.get_train_data(decay=3.)
        model = NPHC(100., alpha=0.9, max_iter=300, print_every=30,
                     step=1e-2, solver='adam', verbose=True)
        model.fit(timestamps)
        R_pred = model.solve()

        expected_R_pred = [[0.423305, -0.559607, -0.307212],
                           [-0.30411, 0.27066, -0.347162],
                           [0.484648, 0.331057, 1.591584]]

        np.testing.assert_array_almost_equal(R_pred, expected_R_pred)

        expected_baseline = [36.808583, 32.304106, -15.123118]

        np.testing.assert_array_almost_equal(model.baseline,
                                             expected_baseline)

        expected_adjacency = [[-3.34742247, -6.28527387, -2.21012092],
                              [-2.51556256, -5.55341413, -1.91501755],
                              [1.84706793, 3.2770494, 1.44302449]]

        np.testing.assert_array_almost_equal(model.adjacency,
                                             expected_adjacency)

    def test_hawkes_nphc_cumulants_solve_l1(self):
        timestamps, baseline, adjacency = Test.get_train_data(decay=3.)
        model = NPHC(100., alpha=0.9, max_iter=300, print_every=30,
                     step=1e-2, solver='adam', verbose=True, penalty='l1', C=1)
        model.fit(timestamps)
        R_pred = model.solve()

        expected_R_pred = [[0.434197, -0.552021, -0.308883],
                           [-0.299366, 0.272764, -0.347764],
                           [0.48448, 0.331059, 1.591587]]

        np.testing.assert_array_almost_equal(R_pred, expected_R_pred)

        expected_baseline = [32.788801, 29.324684, -13.275885]

        np.testing.assert_array_almost_equal(model.baseline,
                                             expected_baseline)

        expected_adjacency = [[-2.925945, -5.54899, -1.97438],
                              [-2.201373, -5.009153, -1.740234],
                              [1.652958, 2.939054, 1.334677]]

        np.testing.assert_array_almost_equal(model.adjacency,
                                             expected_adjacency)

    def test_hawkes_nphc_cumulants_solve_l2(self):
        timestamps, baseline, adjacency = Test.get_train_data(decay=3.)
        model = NPHC(100., alpha=0.9, max_iter=300, print_every=30,
                     step=1e-2, solver='adam', verbose=True, penalty='l2',
                     C=0.1)
        model.fit(timestamps)
        R_pred = model.solve()

        expected_R_pred = [[0.516135, -0.484529, -0.323191],
                           [-0.265853, 0.291741, -0.35285],
                           [0.482819, 0.331344, 1.591535]]

        np.testing.assert_array_almost_equal(R_pred, expected_R_pred)

        expected_baseline = [17.066997, 17.79795, -6.07811]

        np.testing.assert_array_almost_equal(model.baseline,
                                             expected_baseline)

        expected_adjacency = [[-1.310854, -2.640152, -1.054596],
                              [-1.004887, -2.886297, -1.065671],
                              [0.910245, 1.610029, 0.913469]]

        np.testing.assert_array_almost_equal(model.adjacency,
                                             expected_adjacency)


if __name__ == "__main__":
    unittest.main()
