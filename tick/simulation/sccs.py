# License: BSD 3 clause

from operator import itemgetter
import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from .base.simu import Simu
from tick.preprocessing.longitudinal_features_lagger\
    import LongitudinalFeaturesLagger
from tick.simulation import SimuHawkesExpKernels, SimuHawkesMulti
from itertools import combinations
from copy import deepcopy


class SimuSCCS(Simu):
    _const_attr = [
        # user defined parameters
        '_exposure_type',
        '_outcome_distribution',
        '_left_cut',
        '_censoring_prob',
        '_censoring_scale',  # redundant with prob ?
        '_batch_size',
        '_distribution',
        # user defined or computed attributes
        '_hawkes_exp_kernel',
        '_coeffs',
        '_time_drift'
    ]

    _attrinfos = {key: {'writable': False} for key in _const_attr}
        
    def __init__(self, n_cases, n_intervals, n_features, n_lags,
                 time_drift=None, exposure_type="single_exposure",
                 distribution="multinomial", sparse=True,
                 censoring_prob=0, censoring_scale=None,
                 coeffs=None, hawkes_exp_kernels=None, n_correlations=0,
                 batch_size=None, seed=None, verbose=True,
                 ):
        super(SimuSCCS, self).__init__(seed, verbose)
        self.n_cases = n_cases
        self.n_intervals = n_intervals
        self.n_features = n_features
        self.n_lags = n_lags
        self.sparse = sparse

        # attributes with restricted value range
        self._exposure_type = None
        self.exposure_type = exposure_type

        self._distribution = None
        self.distribution = distribution

        self._left_cut = 0
        # self.left_cut = features_left_cut  # TODO later : implementation + property

        self._censoring_prob = 0
        self.censoring_prob = censoring_prob

        self._censoring_scale = None
        self.censoring_scale = censoring_scale if censoring_scale \
            else n_intervals / 4

        self._coeffs = None
        self.coeffs = coeffs

        self._batch_size = None
        self.batch_size = batch_size

        # TODO later: add properties for these parameters
        self.n_correlations = n_correlations
        self.hawkes_exp_kernels = hawkes_exp_kernels
        self._time_drift = None  # TODO later: is this property useful?
        self.time_drift = time_drift  # function(t), used only for the paper, allow to add a baseline

    def simulate(self):
        """
        Launch simulation of the data.

        Returns
        -------
        features : `list` of `numpy.ndarray` or `list` of `scipy.sparse.csr_matrix`,
            list of length n_cases, each element of the list of
            shape=(n_intervals, n_features)
            The list of features matrices.

        labels : `list` of `numpy.ndarray`,
            list of length n_cases, each element of the list of
            shape=(n_intervals,)
            The labels vector
            
        censoring : `numpy.ndarray`, shape=(n_cases,), dtype="uint64"
            The censoring data. This array should contain integers in 
            [1, n_intervals]. If the value i is equal to n_intervals, then there
            is no censoring for sample i. If censoring = c < n_intervals, then 
            the observation of sample i is stopped at interval c, that is, the 
            row c - 1 of the corresponding matrix. The last n_intervals - c rows
            are then set to 0.
        
        coeffs : `numpy.ndarray`, shape=(n_features * (n_lags + 1),)
            The coefficients used to simulate the data.
        """
        return Simu.simulate(self)

    def _simulate(self):
        """ Loop to generate batches of samples until n_cases is reached.
        """
        n_lagged_features = (self.n_lags + 1) * self.n_features
        n_cases = self.n_cases
        if self.coeffs is None:
            self.coeffs = np.random.normal(1e-3, 1.1, n_lagged_features)

        features = []
        censored_features = []
        outcomes = []
        censoring = np.zeros((n_cases,), dtype="uint64")
        cases_count = 0
        while cases_count < n_cases:
            _features, _censored_features, _outcomes, _censoring, _n_samples = \
                self._simulate_batch()

            # TODO later: Warning here if n_cases -> n_cases
            n_new_cases = _n_samples
            c = cases_count
            cases_count += n_new_cases
            n = n_cases - c if cases_count >= n_cases else n_new_cases

            features.extend(_features[0:n])
            censored_features.extend(_censored_features[0:n])
            outcomes.extend(_outcomes[0:n])
            censoring[c:c + n] = _censoring[0:n]

        return features, censored_features, outcomes, censoring, self.coeffs

    def _simulate_batch(self):
        """Simulate a batch of samples, each of which have ticked at least once.
        """
        # TODO later: Warning doc if n_cases -> n_cases
        _features, _n_samples = self.simulate_features(self.batch_size)
        _censored_features = deepcopy(_features)
        _outcomes = self.simulate_outcomes(_features)
        _censoring = np.full((_n_samples,), self.n_intervals,
                             dtype="uint64")
        if self.censoring_prob:
            censored_idx = np.random.binomial(1, self.censoring_prob,
                                              size=_n_samples
                                              ).astype("bool")
            _censoring[censored_idx] -= np.random.poisson(
                lam=self.censoring_scale, size=(censored_idx.sum(),)
                ).astype("uint64")
            _censored_features = self._censor_array_list(_censored_features,
                                                         _censoring)
            _outcomes = self._censor_array_list(_outcomes, _censoring)

            _features, _censored_features, _outcomes, censoring, _ = \
                self._filter_non_positive_samples(_features, _censored_features,
                                                  _outcomes, _censoring)

        return _features, _censored_features, _outcomes, _censoring, _n_samples

    def simulate_features(self, n_samples):
        """Simulates features, either `single_exposure` or
         `multiple_exposures` exposures."""
        if self.exposure_type == "single_exposure":
            features, n_samples = self._sim_single_exposure_exposures()
        elif self.exposure_type == "multiple_exposures":
            sim = self._sim_multiple_exposures_exposures
            features = [sim() for _ in range(n_samples)]
        return features, n_samples

    # We just keep it for the tests now
    # TODO later: need to be improved with Hawkes
    def _sim_multiple_exposures_exposures(self):
        features = np.zeros((self.n_intervals, self.n_features))
        while features.sum() == 0:
            # Make sure we do not generate empty feature matrix
            features = np.random.randint(2,
                                         size=(self.n_intervals, self.n_features),
                                         ).astype("float64")
        if self.sparse:
            features = csr_matrix(features, dtype="float64")
        return features

    def _sim_single_exposure_exposures(self):
        if not self.sparse:
            raise ValueError("'single_exposure' exposures can only be simulated"
                             " as sparse feature matrices")

        if self.hawkes_exp_kernels is None:
            np.random.seed(self.seed)
            decays = .002 * np.ones((self.n_features, self.n_features))
            baseline = 4 * np.random.random(self.n_features) / self.n_intervals
            mult = np.random.random(self.n_features)
            adjacency = mult * np.eye(self.n_features)

            if self.n_correlations:
                comb = list(combinations(range(self.n_features), 2))
                if len(comb) > 1:
                    idx = itemgetter(*np.random.choice(range(len(comb)),
                                                       size=self.n_correlations,
                                                       replace=False))
                    comb = idx(comb)

                for i, j in comb:
                    adjacency[i, j] = np.random.random(1)

            self._set('hawkes_exp_kernels', SimuHawkesExpKernels(
                adjacency=adjacency, decays=decays, baseline=baseline,
                verbose=False, seed=self.seed))

        self.hawkes_exp_kernels.adjust_spectral_radius(.1)  # TODO later: allow to change this parameter
        hawkes = SimuHawkesMulti(self.hawkes_exp_kernels,
                                 n_simulations=self.n_cases)

        run_time = self.n_intervals
        hawkes.end_time = [1 * run_time for _ in range(self.n_cases)]
        dt = 1
        self.hawkes_exp_kernels.track_intensity(dt)
        hawkes.simulate()

        # TODO later: using -1 here is hack. Do something better.
        features = [[np.min(np.floor(f)) if len(f) > 0 else -1
                     for f in patient_events]
                    for patient_events in hawkes.timestamps]

        features = [self.to_coo(feat, (run_time, self.n_features)) for feat in
                    features]

        # Make sure patients have at least one exposure?
        exposures_filter = itemgetter(*[i for i, f in enumerate(features)
                                        if f.sum() > 0])
        features = exposures_filter(features)
        n_samples = len(features)

        return features, n_samples

    def simulate_outcomes(self, features):
        features, _, _ = LongitudinalFeaturesLagger(n_lags=self.n_lags).\
            fit_transform(features)

        if self.distribution == "poisson":
            # TODO later: add self.max_n_events to allow for multiple outcomes
            # In this case, the multinomial simulator should use this arg too
            outcomes = self._simulate_poisson_outcomes(features, self.coeffs)
        else:
            outcomes = self._simulate_multinomial_outcomes(features,
                                                           self.coeffs)
        return outcomes

    def _simulate_multinomial_outcomes(self, features, coeffs):
        baseline = np.zeros(self.n_intervals)
        if self._time_drift is not None:
            baseline = self._time_drift(np.arange(self.n_intervals))
        dot_products = [baseline + feat.dot(coeffs) for feat in features]

        def sim(dot_prod):
            dot_prod -= dot_prod.max()
            probabilities = np.exp(dot_prod) / np.sum(np.exp(dot_prod))
            outcomes = np.random.multinomial(1, probabilities)
            return outcomes.astype("int32")

        return [sim(dot_product) for dot_product in dot_products]

    def _simulate_poisson_outcomes(self, features, coeffs,
                                   first_tick_only=True):
        baseline = np.zeros(self.n_intervals)
        if self._time_drift is not None:
            baseline = self._time_drift(np.arange(self.n_intervals))
        dot_products = [baseline + feat.dot(coeffs) for feat in features]

        def sim(dot_prod):
            dot_prod -= dot_prod.max()
            intensities = np.exp(dot_prod)
            ticks = np.random.poisson(lam=intensities)
            if first_tick_only:
                first_tick_idx = np.argmax(ticks > 0)
                y = np.zeros_like(intensities)
                if ticks.sum() > 0:
                    y[first_tick_idx] = 1
            else:
                y = ticks
            return y.astype("int32")

        return [sim(dot_product) for dot_product in dot_products]

    @staticmethod
    def _censor_array_list(array_list, censoring):
        """Apply censoring to a list of array-like objects. Works for 1-D or 2-D
        arrays, as long as the first axis represents the time.
        
        Parameters
        ----------
        array_list : list of numpy.ndarray or list of scipy.sparse.csr_matrix,
            list of length n_cases, each element of the list of
            shape=(n_intervals, n_features) or shape=(n_intervals,)
            The list of features matrices.
            
        censoring : `numpy.ndarray`, shape=(n_cases, 1), dtype="uint64"
            The censoring data. This array should contain integers in 
            [1, n_intervals]. If the value i is equal to n_intervals, then there
            is no censoring for sample i. If censoring = c < n_intervals, then 
            the observation of sample i is stopped at interval c, that is, the 
            row c - 1 of the correponding matrix. The last n_intervals - c rows
            are then set to 0.

        Returns
        -------
        output : `[numpy.ndarrays]`  or `[csr_matrices]`, shape=(n_intervals, n_features)
            The list of censored features matrices.
        
        """
        def censor(array, censoring_idx):
            if sps.issparse(array):
                array = array.tolil()
                array[int(censoring_idx):] = 0
                array = array.tocsr()
            else:
                array[int(censoring_idx):] = 0
            return array

        return [censor(l, censoring[i]) for i, l in enumerate(array_list)]

    # TODO later: replace this method by LongitudinalSamplesFilter preprocessor
    @staticmethod
    def _filter_non_positive_samples(features, features_censored, labels,
                                     censoring):
        """Filter out samples which don't tick in the observation window.

        Parameters
        ----------
        features : list of numpy.ndarray or list of scipy.sparse.csr_matrix,
            list of length n_cases, each element of the list of
            shape=(n_intervals, n_features)
            The list of features matrices.

        labels : list of numpy.ndarray of length n_cases,
            shape=(n_intervals,)
            The list of labels matrices.
        """
        nnz = [np.nonzero(arr)[0] for arr in labels]
        positive_sample_idx = [i for i, arr in enumerate(nnz) if
                               len(arr) > 0]
        if len(positive_sample_idx) == 0:
            raise ValueError("There should be at least one positive sample per\
             batch. Try to increase batch_size.")
        pos_samples_filter = itemgetter(*positive_sample_idx)
        return list(pos_samples_filter(features)), \
               list(pos_samples_filter(features_censored)), \
               list(pos_samples_filter(labels)), \
               censoring[positive_sample_idx], \
               np.array(positive_sample_idx, dtype="uint64")

    @staticmethod
    def to_coo(feat, shape):
        feat = np.array(feat)
        cols = np.where(feat >= 0)[0]
        rows = np.array(feat[feat >= 0])
        if len(cols) == 0:
            cols = np.random.randint(0, shape[1], 1)
            rows = np.random.randint(0, shape[0], 1)
        data = np.ones_like(cols)
        return csr_matrix((data, (rows, cols)), shape=shape, dtype="float64")

    @property
    def exposure_type(self):
        return self._exposure_type

    @exposure_type.setter
    def exposure_type(self, value):
        if value not in ["single_exposure", "multiple_exposures"]:
            raise ValueError("exposure_type can be only 'single_exposure' or "
                             "'multiple_exposures'.")
        self._set("_exposure_type", value)

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, value):
        if value not in ["multinomial", "poisson"]:
            raise ValueError("distribution can be only 'multinomial' or "
                             "'poisson'.")
        self._set("_distribution", value)

    @property
    def censoring_prob(self):
        return self._censoring_prob

    @censoring_prob.setter
    def censoring_prob(self, value):
        if value < 0 or value > 1:
            raise ValueError("censoring_prob value should be in [0, 1].")
        self._set("_censoring_prob", value)

    @property
    def censoring_scale(self):
        return self._censoring_scale

    @censoring_scale.setter
    def censoring_scale(self, value):
        if value < 0:
            raise ValueError("censoring_scale should be greater than 0.")
        self._set("_censoring_scale", value)

    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        if value is not None and \
                        value.shape != (self.n_features * (self.n_lags + 1),):
            raise ValueError("Coeffs should be of shape\
             (n_features * (n_lags + 1),)")
        self._set("_coeffs", value)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value is None and self.distribution == "multinomial":
            self._set("_batch_size", self.n_cases)
        elif value is None:
            self._set("_batch_size", int(min(2000, self.n_cases)))
        else:
            self._set("_batch_size", int(value))
        self._set("_batch_size", max(100, self.batch_size))
