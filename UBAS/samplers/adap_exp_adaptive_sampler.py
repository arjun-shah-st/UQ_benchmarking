"""
adap_exp_adaptive_sampler.py
============================
Subclass of `AdaptiveSampler` which scales the width distribution by an adaptive exponent before sampling new points.
"""
import numpy as np
from UBAS.samplers.adaptive_sampler import AdaptiveSampler
from numpy.typing import NDArray


class AdapExpAdaptiveSampler(AdaptiveSampler):
    """
    Adaptive sampler with an adaptive width exponent
    """
    def __init__(self, directory, dimension, surrogate, generator, bounds, n_iterations, n_batch_points,
                 initial_inputs, initial_targets, test_inputs=None, test_targets=None, scaler = None,
                 intermediate_training=False,
                 plotter=None, save_interval=5, mean_relative_error=False, adaptive_batch_size=False,
                 n_p_samples=10000, width_scaling='linear', starting_exponent=1, mode="min_variance", learning_rate=0.1,
                 momentum_decay=0, adaptive_exponent_method="mom", max_step=10, min_exp=1, max_exp=100):
        """
        Class Initialization. Check AdaptiveSampler and BaseSampler documentation for parameter descriptions

        Parameters
        ----------
        starting_exponent : int default=1
            The initial value of the probability distribution exponent
        learning_rate : float default=0.1
            The learning rate for stochastic gradient descent with momentum used to set the exponent
        momentum_decay : float default=0
            The decay parameter for stochastic gradient descent with momentum used to set the exponent
        adaptive_exponent_method : str default="mom"
            The method for performing adaptive exponentiation. The default is "mom" for momentum and is the only one
            currently supported.
        max_step : float default=10
            The maximum increase in exponent for a single iteration.
        min_exp : float default=1
            The minimum exponent. If an exponent below this is reached, the exponent is set to halfway between the
            current exponent and the minimum exponent.
        max_exp : float default=100
            The maximum exponent. If an exponent above this is reached, the new value is halfway between the current
            exponent and the maximum exponent.
        """
        super().__init__(directory, dimension, surrogate, generator, bounds, n_iterations, n_batch_points,
                 initial_inputs, initial_targets, test_inputs, test_targets, scaler, intermediate_training,
                 plotter, save_interval, mean_relative_error, adaptive_batch_size, n_p_samples, width_scaling,
                 starting_exponent, mode)

        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.max_step = max_step
        self.min_exp = min_exp
        self.max_exp = max_exp
        self._previous_exponent = None
        self._prev_y_preds = None
        self._prev_err_ratio = None

        self.supported_adaptive_methods = {"mom": self._momentum}
        self.adaptive_method = self.supported_adaptive_methods[adaptive_exponent_method]
        self.init_step = 1.2

    def sampling_step(self, n_batch_points) -> NDArray:
        """
        Override of sampling_step method in AdaptiveSampler to save the data necessary for adaptive exponent
        generation.

        Parameters
        ----------
        n_batch_points : int
            The number of points to sample in one call of sampling_step

        Returns
        -------
        new_x : NDArray
            The new adaptively sampled inputs
        """
        new_x = super().sampling_step(n_batch_points)
        self._update_exp(new_x)
        return new_x

    def _update_exp(self, new_x):
        """
        A method which updates the exponent to raise the pmf to

        Parameters
        ----------
        new_x : NDArray
            The newly sampled x values, used as a validation set for the current surrogate
        """

        self.exponent = self.exponent + self.adaptive_method()
        if self.exponent > self.max_exp:
            self.exponent = 1/2 * (self.max_exp - self._previous_exponent) + self._previous_exponent
        if self.exponent < self.min_exp:
            self.exponent = -1/2 * (self._previous_exponent - self.min_exp) + self._previous_exponent
        print(f"{self.exponent=}")
        self._prev_y_preds = self.predict(new_x, **self.predict_kwargs)[0]

    def _momentum(self) -> float:
        """
        A method which returns the update rule for the exponent based on stochastic gradient descent with momentum

        Returns
        -------
        update : float
            The value which is added to the current exponent to form the new exponent.
        """
        i = self._iteration

        if i == 0:
            self._previous_exponent = self.exponent
            return (self.init_step - 1) * self.exponent

        initial_points = self.n_initial_points
        batch_points = self.n_batch_points
        start_index = initial_points + (i - 1) * batch_points
        new_x = self.x_exact[start_index:start_index + batch_points]
        new_y_exact = self.y_exact[start_index:start_index + batch_points]
        new_y_preds = self.predict(new_x, **self.predict_kwargs)[0]

        err_ratio = np.mean(np.abs((new_y_exact - new_y_preds) / (new_y_exact - self._prev_y_preds)))

        if i == 1:
            self._previous_exponent = self.exponent
            self._prev_err_ratio = err_ratio
            return (self.init_step - 1) * self.exponent

        d_loss = err_ratio - self._prev_err_ratio
        d_exp = self.exponent - self._previous_exponent

        self._previous_exponent = self.exponent
        self._prev_err_ratio = err_ratio

        update_rule = self.momentum_decay * d_exp - self.learning_rate * d_loss / d_exp

        if np.abs(update_rule) > self.max_step:
            update_rule = np.sign(update_rule) * self.max_step
        return update_rule
