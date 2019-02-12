"""Spectral functions."""
import logging

import numpy as np

from mne.time_frequency import tfr_multitaper


logger = logging.getLogger('brainets')

# def mt_config(name, freqs):
#     """Get multitaper input parameters according to a config.

#     Parameters
#     ----------
#     name : string
#         Configuration name.
#     """
#     if name == 'julien':
#         is_low = freqs <= 30.
#         n_cycles = np.zeros_like(freqs)
#         n_cycles[is_low] = freqs[is_low] / 6.
#         n_cycles[~is_low] = 0.2 * freqs[~is_low]
#         time_bandwidth = 4.
#     return n_cycles,


def mt_hga(epoch, f=100., n_cycles=12, time_bandwidth=20., **kw):
    """Extract high-gamma activity (HGA) around a central frequency.

    The HGA is extracted using the `mne.time_frequency.tfr_multitaper`
    function.

    Parameters
    ----------
    epoch : mne.Epochs
        Instance of mne.Epochs
    f : float | 100.
        The central high-gamma frequency
    n_cycles : int | 12
        The number of cycles to use for the frequency resolution.
    time_bandwidth : float | 20.
        Time x (Full) Bandwidth product.
    karg : dict | {}
        Additional arguments are passed to the `tfr_multitaper` function.

    Returns
    -------
    tf : AverageTFR | EpochsTFR
        The averaged or single-trial HGA.
    """
    assert isinstance(f, (int, float))
    freq = np.array([f])
    tf = tfr_multitaper(epoch, freq, freq / n_cycles, return_itc=False,
                        time_bandwidth=time_bandwidth, **kw)
    return tf