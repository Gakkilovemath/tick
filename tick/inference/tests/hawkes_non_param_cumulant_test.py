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

    def test_hawkes_nphc_rectangular(self):
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

        for method in ['classic', 'parallel_by_day', 'parallel_by_component']:
            model = NPHC()
            model.fit(timestamps, method=method, filtr='rectangular')

            np.testing.assert_array_almost_equal(model.L, expected_L)
            np.testing.assert_array_almost_equal(model.C, expected_C)
            np.testing.assert_array_almost_equal(model.K_c, expected_K)


    def test_hawkes_nphc_gaussian(self):
        timestamps, baseline, adjacency = Test.get_train_data(decay=3.)

        expected_L = [[2.18034168, 2.8594481, 4.45869871],
                      [1.99015833, 2.6472106, 4.16933169],
                      [2.27845603, 2.89257894, 4.7639535]]

        expected_C = [[[13.46997993, 12.80439244, 23.38769407],
                       [12.80439244, 16.42386725, 25.51214473],
                       [23.38769407, 25.51214473, 47.48551437]],

                      [[18.25904694, 20.351048, 37.05065869],
                       [20.351048, 25.97551921, 43.21649302],
                       [37.05065869, 43.21649302, 81.48058524]],

                      [[7.7311428, 4.59067863, 13.75166764],
                       [4.59067863, 10.06180264, 14.23649572],
                       [13.75166764, 14.23649572, 37.35567706]]]

        expected_K = [
            np.array([[268.34025438, -16.7512349, 589.79949575],
                      [134.076714, -185.32556299, 207.679562],
                      [398.81411791, -109.17929498, 1124.91155354]]),
            np.array([[57.44595962, -107.46285564, 627.05718534],
                      [-16.88754034, -224.42590084, 455.44475719],
                      [192.44204653, -98.24137747, 1879.98149655]]),
            np.array([[283.34149916, 271.60393925, 428.00189363],
                      [291.72437524, 306.74765368, 173.30696925],
                      [384.77307656, 246.61404879, 401.65027457]])]

        for method in ['classic']:
            model = NPHC()
            model.fit(timestamps, method=method, filtr='gaussian')

            np.testing.assert_array_almost_equal(model.L, expected_L)
            np.testing.assert_array_almost_equal(model.C, expected_C)
            np.testing.assert_array_almost_equal(model.K_c, expected_K)


if __name__ == "__main__":
    unittest.main()
