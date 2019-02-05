"""Spectral functions."""
import numpy as np

from mne.time_frequency import tfr_multitaper


def mt_gamma(epoch, f=100., n_cycles=12, time_bandwidth=20., **kw):
    """Extract gamma activity around a central frequency.

    The gamma is extracted using the `mne.time_frequency.tfr_multitaper`
    function.

    Parameters
    ----------
    epoch : mne.Epochs
        Instance of mne.Epochs
    f : float | 100.
        The central gamma frequency
    n_cycles : int | 12
        The number of cycles to use for the frequency resolution.
    time_bandwidth : float | 20.
        Time x (Full) Bandwidth product.
    karg : dict | {}
        Additional arguments are passed to the `tfr_multitaper` function.
    """
    assert isinstance(f, (int, float))
    freq = np.array([f])
    tf = tfr_multitaper(epoch, freq, freq / n_cycles, return_itc=False,
                        time_bandwidth=time_bandwidth, **kw)
    return tf
