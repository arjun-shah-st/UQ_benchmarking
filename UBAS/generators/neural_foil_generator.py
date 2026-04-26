"""
neural_foil_generator.py
========================
A function that runs neuralfoil on combinations of input parameters
"""
import numpy as np
import neuralfoil as nf
from UBAS.generators.base_generator import BaseGenerator
from typing import Tuple
from numpy.typing import NDArray
import tqdm

class NFGenerator(BaseGenerator):
    def __init__(self, qoi="CM", airfoil=None):
        """
        Initialization

        Parameters
        ----------
        qoi : str default="CM"
            The quantity of interest from neuralfoil. Currently supported values are "CL", "CD", and "CM"
        airfoil : NDArray default=None
            If None, data is generated for a five dimensional input `x`, which includes NACA parameters.
            Otherwise, an NDArray should be given to describe the airfoil of form:
                [Maximum camber to chord ratio,
                Location of Maximum Camber in terms of x/c
                Maximum Thickness to chord ratio]
            Then, data will be generated for the specific airfoil as a function of Reynolds Number and Angle of Attack
        """
        self.qoi = qoi
        if airfoil is not None:
            self._given_airfoil = True
            self._airfoil = airfoil
        else:
            self._given_airfoil = False

    def generate(self, x, *args, **kwargs) -> Tuple[NDArray, NDArray]:
        """
        Returns the inputs `x` and the neuralfoil function value for the parameter inputs

        Parameters
        ----------
        x : NDArray
            The input array for which to generate data.
            If an airfoil is not given in initialization, the shape should be [no_samples, 5], where the indices along
            axis 1 represent:
                0: Maximum Camber to chord ratio
                1: Location of Maximum Camber in terms of x/c
                2: Maximum Thickness to chord ratio
                3: Reynolds Number based on chord
                4: Angle of attack (degrees)
            If an airfoil is given in initialization the shape should be [no_samples, 2], where the indices along axis 1
            are the Reynolds Number and Angle of attack
        *args
            Extra arguments should be passed as keyword arguments
        **kwargs
            Extra keyword arguments which are passed to neuralfoil

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
        if self._given_airfoil:
            complete_x = np.c_[self._airfoil * np.ones((x.shape[0], 3)), x]
        else:
            complete_x = x

        y = np.empty(complete_x.shape[0])

        for i, x_val in enumerate(tqdm.tqdm(complete_x)):
            coords = create_naca_4_series(x_val[0], x_val[1], x_val[2])
            data = nf.get_aero_from_coordinates(coordinates=coords, Re=x_val[3], alpha=x_val[4], model_size='xxxlarge', **kwargs)
            y[i] = data[self.qoi]
        return x, y


def create_naca_4_series(e, p, t, n=200):
    """Helper function for generating NACA airfoils"""
    x = np.linspace(0, np.pi, n)
    x = 1/2 * (1-np.cos(x))
    camber = np.where(x<p, e * x/p ** 2 * (2*p-x), e*(1-x)/(1-p)**2 * (1 + x - 2 * p))
    thickness = 10 * t * (0.2969 * x ** 0.5 - 0.126 * x - 0.3537 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    top_surface = camber + 1/2 * thickness
    bottom_surface = camber - 1/2 * thickness
    top_surface = top_surface[::-1]
    coordinates = np.append(top_surface[:-1], bottom_surface[:-1], axis=0)
    x = np.append(x[::-1][:-1], x[:-1])

    return np.c_[x, coordinates]


