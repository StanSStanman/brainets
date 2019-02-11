"""Utility functions."""
import numpy as np

def normalize(x, to_min=0., to_max=1.):
    """Normalize the array x between tomin and tomax.

    Parameters
    ----------
    x : array_like
        The array to normalize
    to_min : int/float | 0.
        Minimum of returned array
    to_max : int/float | 1.
        Maximum of returned array

    Returns
    -------
    xn : array_like
        The normalized array
    """
    x = np.asarray(x)
    x_min, x_max = x.min(), x.max()
    assert x_min != x_max, "Not working for constant arrays"
    return (to_max - to_min) * (x - x_min) / (x_max - x_min) + to_min
