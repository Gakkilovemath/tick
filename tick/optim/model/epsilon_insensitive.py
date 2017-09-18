# License: BSD 3 clause

import numpy as np
from .base import ModelGeneralizedLinear, ModelFirstOrder, ModelLipschitz
from .build.model import ModelEpsilonInsensitive as _ModelEpsilonInsensitive

__author__ = 'Stephane Gaiffas'


class ModelEpsilonInsensitive(ModelFirstOrder,
                              ModelGeneralizedLinear):
    """Hinge loss model for binary classification. This class gives first order
    information (gradient and loss) for this model.

    Parameters
    ----------
    fit_intercept : `bool`
        If `True`, the model uses an intercept

    Attributes
    ----------
    features : `numpy.ndarray`, shape=(n_samples, n_features) (read-only)
        The features matrix

    labels : `numpy.ndarray`, shape=(n_samples,) (read-only)
        The labels vector

    n_samples : `int` (read-only)
        Number of samples

    n_features : `int` (read-only)
        Number of features

    n_coeffs : `int` (read-only)
        Total number of coefficients of the model

    n_threads : `int`, default=1 (read-only)
        Number of threads used for parallel computation.

        * if ``int <= 0``: the number of physical cores available on
          the CPU
        * otherwise the desired number of threads
    """

    _attrinfos = {
        "smoothness": {
            "writable": True,
            "cpp_setter": "set_threshold"
        }
    }

    def __init__(self, fit_intercept: bool = True, threshold: float = 1,
                 n_threads: int = 1):
        ModelFirstOrder.__init__(self)
        ModelGeneralizedLinear.__init__(self, fit_intercept)
        self.n_threads = n_threads
        self.threshold = threshold

    # TODO: implement _set_data and not fit
    def fit(self, features, labels):
        """Set the data into the model object

        Parameters
        ----------
        features : `numpy.ndarray`, shape=(n_samples, n_features)
            The features matrix

        labels : `numpy.ndarray`, shape=(n_samples,)
            The labels vector

        Returns
        -------
        output : `ModelLogReg`
            The current instance with given data
        """
        ModelFirstOrder.fit(self, features, labels)
        ModelGeneralizedLinear.fit(self, features, labels)
        ModelLipschitz.fit(self, features, labels)
        self._set("_model", _ModelEpsilonInsensitive(self.features,
                                                     self.labels,
                                                     self.fit_intercept,
                                                     self.threshold,
                                                     self.n_threads))
        return self

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        self._model.grad(coeffs, out)

    def _loss(self, coeffs: np.ndarray) -> float:
        return self._model.loss(coeffs)
