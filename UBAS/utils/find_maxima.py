"""
find_maxima.py
==============
A simple utility to return the predicted local maxima of a surrogate model using a basin-hopping approach
"""
import numpy as np
from scipy.optimize import basinhopping
from typing import Tuple


def find_local_maxima(surrogate, bounds, n_extrema = 1, mode='maximize'):
    tup_bounds = tuple(tuple(inner_list) for inner_list in list(bounds.T))
    ndim = bounds.shape[1]

    def invert(x):
        pred_results = surrogate.predict(x)
        if isinstance(pred_results, Tuple):
            y_preds, y_bounds = pred_results
        else:
            y_bounds = pred_results
            y_preds = np.mean(pred_results, axis=1)
        if mode == 'maximize':
            return -1 * surrogate.predict(y_preds)
        elif mode == 'minimize':
            return surrogate.predict(y_preds)
    scaling_factor = bounds[1] - bounds[0]
    additive_factor = bounds[0]

    rng = np.random.default_rng()
    init_input = rng.random(ndim) * scaling_factor + additive_factor

    extrema = np.ones(n_extrema) * np.inf
    locations = np.empty((n_extrema, ndim))

    def callback_fun(x, f, accepted):
        min_extrema = np.min(extrema)
        argmin_extrema = np.argmin(extrema)
        if f < min_extrema:
            extrema[argmin_extrema] = f
            locations[argmin_extrema] = x

    ret = basinhopping(invert, init_input, minimizer_kwargs={"bounds": tup_bounds}, niter=10, callback=callback_fun,
                       rng=np.random.default_rng())
    if mode == 'maximize':
        extrema = -1 * extrema
    return extrema, locations

