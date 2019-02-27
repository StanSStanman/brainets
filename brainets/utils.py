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


def moving_average(arr, kern_size, axis=0, use_fft=True):
    """Perform a moving average for multi-dimentional data.

    The moving average can be used to smooth a signal. It perform a simple
    convolution with a stable kernel (e.g. [.25, .25, .25, .25])

    Parameters
    ----------
    arr : array_like
        Array to smooth
    kern_size : int
        Number of points to use for the kernel
    axis : int | 0
        Axis along which the moving average is performed
    use_fft : bool | True
        Use a fft based convolution (scipy.signal.fftconvolve) or not
        (np.convolve)

    Returns
    -------
    arr_sm : array_like
        The smoothed array with the same shape as `arr`
    """
    assert isinstance(kern_size, int) and isinstance(arr, np.ndarray)
    # Construct the kernel
    kernel = np.ones((kern_size,), dtype=float) / kern_size
    # fft based convolution
    if use_fft:
        from scipy.signal import fftconvolve
        fcn = fftconvolve
    else:
        fcn = np.convolve
    return np.apply_along_axis(lambda m: fcn(m, kernel, mode='same'),
                               axis=axis, arr=arr)
