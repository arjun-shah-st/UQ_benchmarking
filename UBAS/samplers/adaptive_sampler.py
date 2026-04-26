"""
adaptive_sampler.py
===================
Subclass of `BaseSampler`, and is a base class for all types of adaptive samplers.
The important change is the reimplementation of the sample_step() function which determines how new points are
generated.
"""
import numpy as np
from UBAS.samplers.base_sampler import BaseSampler
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt

class AdaptiveSampler(BaseSampler):
    """Adaptive Sampling Parent Class"""
    def __init__(self, directory, dimension, surrogate, generator, bounds, n_iterations, n_batch_points,
                 initial_inputs, initial_targets, test_inputs=None, test_targets=None, scaler=None,
                 intermediate_training=False,
                 plotter=None, save_interval=5, mean_relative_error=False, adaptive_batch_size=False,
                 n_p_samples=10000, width_scaling='linear', starting_exponent=1, mode="min_variance"):
        """
        Class Initialization. Check BaseSampler documentation for parameter descriptions

        Parameters
        ----------
        n_p_samples : int default=10000
            The number of samples uniformly generated within the bounds to approximate the probability distribution
            from which samples are drawn.
        width_scaling : str default='linear'
            The method by which the widths are scaled before being transformed into a probability distribution
            Currently supported values: 'linear', 'logistic'
            If linear, an affine transformation is used to map the widths to the interval [0, 1]
            If logistic, an affine transformation maps the widths to [-6, 6] and is then transformed
                by the standard logistic function.
        mode : str default='min_variance'
            The mode of the adaptive sampler. If min_variance, it will use the PMF of the widths to attempt to minimize
            uncertainty across the domain. If max, it will sample from the PMF defined by the upper bound of the
            predictions in an attempt to resolve local maxima.
        """
        super().__init__(directory, dimension, surrogate, generator, bounds, n_iterations, n_batch_points,
                         initial_inputs, initial_targets, test_inputs, test_targets, scaler,
                         intermediate_training, plotter, save_interval, mean_relative_error, adaptive_batch_size)
        self.n_p_samples = n_p_samples
        supported_width_scaling = {"linear": AdaptiveSampler.scale_values_linear,
                                   "logistic": AdaptiveSampler.scale_values_logistic}
        self.width_scaling_method = supported_width_scaling[width_scaling]
        self.exponent = starting_exponent
        self.mode = mode

    def sampling_step(self, n_batch_points) -> NDArray:
        """
        Override of sampling_step method in BaseSampler which uses the surrogate to predict `n_p_samples` values
        and draw from the PMF proportional to their widths if mode is "min_variance", or from the PMF proportional to
        the predicted upper-bound if the mode is "max".

        Parameters
        ----------
        n_batch_points : int
            The number of points to sample in one call of sampling_step

        Returns
        -------
        new_x : NDArray
            The new adaptively sampled inputs
        """
        exponent = self.exponent
        probability_eval_points = self.sampler.uniformly_sample(self.n_p_samples)
        print("Sampling probability distribution...")
        p_preds, p_bounds = self.predict(probability_eval_points)
        print (probability_eval_points)
        if self.mode == "min_variance":
            pmf = p_bounds[:, 1] - p_bounds[:, 0]
            p = np.where(pmf < 0, 0, pmf)
        elif self.mode == "max":
            pmf = p_bounds[:, 1]
            p = pmf - np.min(pmf)
        print(p)
        p = self.width_scaling_method(p)
        p = p ** exponent / np.sum(p ** exponent)

        new_x = self.sampler.adaptively_mc_sample(probability_eval_points, p, n_batch_points)
        return new_x

    @staticmethod
    def scale_values_linear(values):
        """Linear scaling of values to the interval [0, 1]"""
        min_val = np.min(values)
        max_val = np.max(values)
        return (values - min_val) / (max_val - min_val)

    @staticmethod
    def scale_values_logistic(values):
        """Logistic scaling of values to the interval (0, 1)"""
        LOGISTIC_LB = -6
        LOGISTIC_UB = 6
        linear_values = AdaptiveSampler.scale_values_linear(values) * (LOGISTIC_UB - LOGISTIC_LB) + LOGISTIC_LB
        return 1 / (1 + np.exp(-linear_values))
