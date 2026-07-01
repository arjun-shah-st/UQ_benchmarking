"""
borehole_generator.py
===============================
Data generator for the borehole benchmark function.
"""
from typing import Tuple
from numpy.typing import NDArray
from UBAS.generators.base_generator import BaseGenerator
import numpy as np


class BoreholeGenerator(BaseGenerator):
    """Generator class for the 8-dimensional borehole function benchmark."""

    def generate(self, x, *args, **kwargs) -> Tuple[NDArray, NDArray]:
        """Evaluate the borehole function for a batch of inputs.

        Parameters
        ----------
        x : NDArray
            Input array of shape (n_samples, 8), where each column is one of:
            [r_w, r, T_u, H_u, T_l, H_l, L, K_w].

        Returns
        -------
        x : NDArray
            The input array passed through.
        y : NDArray
            The borehole function outputs for each input row.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.ndim != 2 or x.shape[1] != 8:
            raise ValueError("BoreholeGenerator requires input shape (n_samples, 8)")

        rw = x[:, 0]
        r = x[:, 1]
        Tu = x[:, 2]
        Hu = x[:, 3]
        Tl = x[:, 4]
        Hl = x[:, 5]
        L = x[:, 6]
        Kw = x[:, 7]

        numerator = 2.0 * np.pi * Tu * (Hu - Hl)
        log_term = np.log(r / rw)
        denominator = log_term * (
            1.0
            + (2.0 * L * Tu) / (log_term * rw**2 * Kw)
            + Tu / Tl
        )

        y = numerator / denominator
        return x, y
