"""
central_peak_generator.py
==============================
Data generator for the central peak function
"""
from typing import Tuple
from numpy.typing import NDArray
from UBAS.generators.base_generator import BaseGenerator
import numpy as np


class CentralPeakGenerator(BaseGenerator):
    """Generator class for the central peak function in arbitrary dimensions"""

    def __init__(self, exp=-10):
        """
        Initialization for Central Peak Generator

        Parameters
        ----------
        exp : float default = -10
            The front factor of the norm of the input vector which e is raised to
        """
        self.exp = exp

    def generate(self, x, *args, **kwargs) -> Tuple[NDArray, NDArray]:
        """
        Returns the inputs `x` and the central peak function values for inputs `x.

        Parameters
        ----------
        x : NDArray
            The inputs for which to generate data. The shape
            should be [no_samples, no_dimensions]
        *args
            Extra arguments to `generate` should be passed as keyword arguments
        **kwargs
            Extra arguments to `generate`: refer to each generator documentation for
            a list of all possible arguments.

        Returns
        -------
        generated_x : NDArray
            The inputs corresponding to the generated data
            Should be the same shape as the parameter `x`.
        generated_Y : 1DArray
            The outputs corresponding to the generated data
            0-axis should have the same dimension as the 0-axis
            of `generated_x`
        """
        return x, np.e ** (self.exp * np.linalg.norm(x, axis=1) ** 2)
