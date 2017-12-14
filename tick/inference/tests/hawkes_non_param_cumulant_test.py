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

        self.model = NPHC()

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

    def test_hawkes_nphc(self):
        timestamps, baseline, adjacency = Test.get_train_data(decay=3.)
        self.model.fit(timestamps, method='classic')

        expected_L = [[2.18111273, 2.86045931, 4.46027548],
                      [1.99473086, 2.65329275, 4.17891103],
                      [2.27958416, 2.89401114, 4.76631226]]
        np.testing.assert_array_almost_equal(self.model.L, expected_L)

        expected_C = [[[13.0930694, 9.810038, 13.98120564],
                       [9.810038, 10.43193551, 7.32150065],
                       [13.98120564, 7.32150065, 3.06155434]],

                      [[30.68701512, 40.28176486, 62.1292852],
                       [40.28176486, 53.83301226, 83.89491855],
                       [62.1292852, 83.89491855, 130.48244664]],

                      [[1.42181832, -1.61459159, 10.69659146],
                       [-1.61459159, 3.77494657, 13.42391358],
                       [10.69659146, 13.42391358, 57.13355973]]]

        np.testing.assert_array_almost_equal(self.model.C, expected_C)

        expected_K = [
            np.array([[3302.26643679, 3109.40412929, 20679.52098654],
                      [3221.39899239, 2948.21588706, 24809.32377033],
                      [8072.89275955, 10053.50101058, 53687.97529961]]),
            np.array([[-5117.61774403, -9240.87205519, -21671.41368118],
                      [-6891.15426125, -12349.6640054, -29279.33819162],
                      [-10560.22923388, -19123.76532334, -44208.02921952]]),
            np.array([[2538.81664527, 4262.11108296, 1825.47001737],
                      [3371.94869819, 5067.88151348, 2282.03495219],
                      [3067.82859246, 4863.97068752, -4667.24341696]])]

        np.testing.assert_array_almost_equal(self.model.K_c, expected_K)


if __name__ == "__main__":
    unittest.main()
