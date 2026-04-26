"""
base_generator.py
==============================
Abstract base class for data generators; i.e., functions which take
some N-dimensional input and return a one dimensional output
"""
from abc import ABC, abstractmethod
from typing import Tuple
from numpy.typing import NDArray


class BaseGenerator(ABC):
    """Abstract base class for data generators."""

    @abstractmethod
    def generate(self, x, *args, **kwargs) -> Tuple[NDArray, NDArray]:
        """

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

        pass
