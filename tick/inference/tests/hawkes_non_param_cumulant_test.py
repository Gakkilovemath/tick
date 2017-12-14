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

        expected_L = [[2.18111273, 2.86045931, 4.46027548],
                      [1.99473086, 2.65329275, 4.17891103],
                      [2.27958416, 2.89401114, 4.76631226]]

        expected_C = [[[13.0930694, 9.810038, 13.98120564],
                       [9.810038, 10.43193551, 7.32150065],
                       [13.98120564, 7.32150065, 3.06155434]],

                      [[30.68701512, 40.28176486, 62.1292852],
                       [40.28176486, 53.83301226, 83.89491855],
                       [62.1292852, 83.89491855, 130.48244664]],

                      [[1.42181832, -1.61459159, 10.69659146],
                       [-1.61459159, 3.77494657, 13.42391358],
                       [10.69659146, 13.42391358, 57.13355973]]]

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

        for method in ['classic', 'parallel_by_day', 'parallel_by_component']:
            model = NPHC()
            model.fit(timestamps, method=method, filtr='rectangular')

            np.testing.assert_array_almost_equal(model.L, expected_L)
            np.testing.assert_array_almost_equal(model.C, expected_C)
            np.testing.assert_array_almost_equal(model.K_c, expected_K)

    def test_hawkes_nphc_gaussian(self):
        timestamps, baseline, adjacency = Test.get_train_data(decay=3.)

        expected_L = [[2.18111273, 2.86045931, 4.46027548],
                      [1.99473086, 2.65329275, 4.17891103],
                      [2.27958416, 2.89401114, 4.76631226]]

        expected_C = [[[13.40973172, 12.72343891, 23.26299118],
                       [12.72343891, 16.31727924, 25.34636396],
                       [23.26299118, 25.34636396, 47.23032538]],

                      [[17.94047753, 19.91772422, 36.38105748],
                       [19.91772422, 25.39641497, 42.31134567],
                       [36.38105748, 42.31134567, 80.08839132]],

                      [[7.63870224, 4.47006984, 13.5557646],
                       [4.47006984, 9.90993685, 13.98480317],
                       [13.5557646, 13.98480317, 36.94734504]]]

        expected_K = [
            np.array([[272.87738976, -8.37906733, 609.52324631],
                      [140.31117416, -174.33459879, 234.10699262],
                      [408.34350693, -92.03289127, 1165.14812564]]),
            np.array([[76.70724028, -70.527509, 712.45835703],
                      [10.05629688, -174.52644617, 573.18990427],
                      [233.26512233, -21.07683388, 2056.29391411]]),
            np.array([[290.9400127, 284.75867571, 462.11035356],
                      [301.94149592, 323.05005103, 217.47940958],
                      [401.03337433, 273.76524566, 471.87953342]])]

        for method in ['classic']:
            model = NPHC()
            model.fit(timestamps, method=method, filtr='gaussian')

            np.testing.assert_array_almost_equal(model.L, expected_L)
            np.testing.assert_array_almost_equal(model.C, expected_C)
            np.testing.assert_array_almost_equal(model.K_c, expected_K)


if __name__ == "__main__":
    unittest.main()
