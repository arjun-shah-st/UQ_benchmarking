"""
benchmark_function_generator.py
===============================
An example generator function that returns x,y pairs based on the optimization benchmark functions from opfunu
"""
from UBAS.generators.base_generator import BaseGenerator
from typing import Tuple
from numpy.typing import NDArray
import opfunu.name_based as bench
import numpy as np


class BenchmarkFunctionGenerator(BaseGenerator):
    """
    Generator that evaluates a given SciPy benchmark function on an array of x inputs.
    """

    def __init__(self, function_name: str):
        """
        Parameters
        ----------
        function_name : str
            The name of the scipy benchmark function

        Raises
        ------
        ValueError
            If the function name is not contained within the opfunu class
        """

        try:
            self.function = getattr(bench, function_name)
        except AttributeError:
            raise ValueError(f"The function name: {function_name} was not found in opfunu")
        self.index = 0

    def generate(self, x) -> Tuple[NDArray, NDArray]:
        """
        Generates y-values by evaluating the function on x-values.

        Parameters
        ----------
        x : NDArray
            The `x` values to generate y values for. Should be of shape (n_samples, n_dimensions)

        Returns
        -------
        x : NDArray
            The `x` values that were evaluated
        y_gen : NDArray
            The Y values that were generated
        """
        n_dim = x.shape[1]
        fun = self.function(ndim=n_dim).evaluate
        y_gen = np.zeros(x.shape[0])

        for i, feature in enumerate(x):
            y_gen[i] = fun(feature)
        return x, y_gen
