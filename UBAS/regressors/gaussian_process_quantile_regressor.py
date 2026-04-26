"""
gaussian_process_quantile_regressor.py
======================================
A regressor which uses gaussian process regression to fit data
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.base import BaseEstimator
from UBAS.utils.distance import minmax_dist
from scipy.stats import norm
import numpy as np


class GaussianProcessQuantileRegressor(BaseEstimator):
    def __init__(self, quantiles=[0.05, 0.95]):
        self.quantiles = quantiles
        self.std = norm.ppf(quantiles[1])
        print (self.std)
        self.kernel = None
        self.model = None

    def fit(self, X, y, **fit_params):
        X, y = self._validate_data(X, y, accept_sparse=True)
        min_dist, max_dist = minmax_dist(X)
        min_dist = max(min_dist, 10 ** -4)
        self.kernel = RBF((min_dist + max_dist)/2, (min_dist, max_dist))
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
        self.model.fit(X, y)
        self.kernel = self.model.kernel_
        return self

    def predict(self, X):
        y_pred, sigma = self.model.predict(X, return_std=True)
        return y_pred, np.stack([y_pred-sigma * self.std, y_pred + sigma * self.std], axis=1)
