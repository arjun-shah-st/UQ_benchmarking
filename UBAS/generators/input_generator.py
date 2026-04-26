"""
input_generator.py
==============================
Generates inputs within bounds, either uniformly or based off of a given probability distribution
"""
import numpy as np
from numpy.typing import NDArray


class InputGenerator:
    """Simple class to generate data points within given bounds, either uniformly or adaptively"""

    def __init__(self, bounds, ndim, seed=None):
        """
        Class Initialization

        Parameters
        ----------
        bounds : NDArray
            The bounds within which to sample. Should be an NDArray of shape (2, n_dimensions) where
            the first element of the 0th axis are the lower bounds and the second element are the upper
            bounds for each dimension.
        ndim : int
            The number of dimensions (inputs) of the problem to be sampled
        seed: int = None
            The random seed used for reproducibility

        Raises
        -------
        ValueError
            If the shape of bounds is not (2, ndim)
        ValueError
            If any of the lower bounds (first index of bounds) are larger than any of the upper bounds (second index)
        """
        if bounds.shape != (2, ndim):
            raise ValueError(f"""The detected shape of bounds was {bounds.shape}; it is required that bounds has a
            shape of (2, ndim). The format of bounds is:
            NDArray([[lower_bounds for each dimension], [upper bounds for each dimension]]).""")
        if any(bounds[1] - bounds[0] <= 0):
            raise ValueError(f"""The lower bounds must be strictly smaller than the upper bounds. The lower bounds were: 
            {bounds[0]}, while the upper bounds were {bounds[1]}. The format of bounds is: 
            NDArray([[lower_bounds for each dimension], [upper bounds for each dimension]]).""")

        self.ndim = ndim
        self.bounds = bounds

        # Define transformation variables for transforming range of [0, 1) to the desired bounds
        self._scaling_factor = bounds[1]-bounds[0]
        self._additive_factor = bounds[0]
        self.seed = seed

    def uniformly_sample(self, batch_samples=1) -> NDArray:
        """
        Generates uniform samples within the bounds given in class initialization

        Returns
        -------
        new_x : NDArray
            The new locations of training data which should augment the current training set of
            shape (`batch_samples`, n_dimensions)
        batch_samples : int, default=1
            The number of samples to generate with one function call.
        """
        rng = np.random.default_rng(self.seed)
        new_x = rng.random((batch_samples, self.ndim)) * self._scaling_factor + \
                np.ones((batch_samples, self.ndim)) * self._additive_factor
        return new_x

    def adaptively_mc_sample(self, x_candidates, p, batch_samples=1, bin_width=None) -> NDArray:
        """
        Generates adaptive samples within the bounds given in class initialization using
        a Monte Carlo sampled probability distribution

        Parameters
        ----------
        x_candidates : NDArray
            Locations which are used to sample the probability distribution, `p`
            Should be of shape ((n_samples, ndim))
        p : NDArray
            Samples of the probability distribution from which to generate new x values
            Should be of shape (n_samples)
        batch_samples : int, default=1
            The number of samples to generate with one function call.
        bin_width : Union(float, NDArray, NoneType) default=None
            The side length of the square around each x_candidate that samples are chosen from if x_candidate is drawn
            from `p`. If a float is passed, the bin width is assumed to be constant for each dimension. If an NDArray
            of shape (n_dim) is passed, it is interpreted as the bin widths for each dimension.
            If none, the number of bins per dimension is calculated by taking the number of bins per dimension to the
            power of the inverse of the number of dimensions. This assumes a gridlike distribution.

        Returns
        -------
        new_x : NDArray
            The new locations of training data which should augment the current training set of
            shape (`batch_samples`, n_dimensions)

        Raises
        -------
        ValueError
            If the shape of x_candidates is inconsistent with the number of dimensions
        ValueError
            If x_candidates and p do not have the same length
        ValueError
            If `bin_width` is an NDArray which does not have shape (n_dim)
        """

        if x_candidates.shape[1] != self.ndim:
            raise ValueError(f"""Shape of x_candidates was found to be inconsistent with the number of dimensions 
                             the detected shape was {x_candidates.shape}, implying a dimension of 
                             {x_candidates.shape[1]}, while the number of dimensions is {self.ndim} 
                             please either change sampler.ndim or reshape x_candidates""")

        if p.shape[0] != (x_candidates.shape[0]):
            raise ValueError(f"""Length of probability distribution, `p` inconsistent with number of samples in 
                             x_candidates. The length of `p` is {p.shape[0]}, while the shape of x_candidates is 
                             {x_candidates.shape}""")

        if isinstance(bin_width, np.ndarray) and bin_width.shape[0] != self.ndim:
            raise ValueError(f"""Bin width was passed as an array, but its length does not match `ndim`. The detected 
                             shape was {bin_width.shape}""")

        rng = np.random.default_rng(self.seed)

        # If bin width is none, calculate it assuming a grid_like distribution:
        if bin_width is None:
            bin_width = self._scaling_factor / (x_candidates.shape[0] ** (1/self.ndim))

        # If bin width is zero, sample without replacement
        rep = False if all(bin_width) == 0 else True

        if isinstance(bin_width, np.ndarray) is False:
            bin_width = np.ones(self.ndim) * bin_width

        # Choose indices from probability distribution
        new_x_indices = rng.choice(p.shape[0], batch_samples, p=p, replace=rep)
        chosen_candidates = x_candidates[new_x_indices]

        dx = bin_width / 2
        deviation = rng.uniform(-1, 1, chosen_candidates.shape) * dx
        new_x = chosen_candidates + deviation

        # if any of the x-values are outside the bounds, set their value to the bound
        lb_comparison = self.bounds[0] * np.ones_like(new_x)
        ub_comparison = self.bounds[1] * np.ones_like(new_x)

        new_x = np.maximum(lb_comparison, new_x)
        new_x = np.minimum(ub_comparison, new_x)

        return new_x
