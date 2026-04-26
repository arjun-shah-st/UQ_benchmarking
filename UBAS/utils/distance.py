"""
distance.py
===========
A set of utility functions for calculating distances between points
"""
import numpy as np
from typing import Tuple


def minmax_dist(points) -> Tuple[float, float]:
    """
    A simple function to return the minimum and maximum distances between points in an array

    Parameters
    ----------
    points : NDArray
        The set of points to calculate the minimum and maximum distances for

    Returns
    -------
    min_dist : float
        The minimum distance between any two points in the array
    max_dist : float
        The maximum distance between any two points in the array
    """
    min_dist = float('inf')
    max_dist = float('0')

    for i in range(points.shape[0]):
        for j in range(i+1, points.shape[0]):
            dist = np.linalg.norm(points[i] - points[j])
            min_dist = min(dist, min_dist)
            max_dist = max(dist, max_dist)

    return min_dist, max_dist
