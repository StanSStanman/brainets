"""Spectral functions."""
import logging

import numpy as np

from mne.time_frequency import tfr_multitaper, EpochsTFR


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


def mt_hga_julien(epoch, f=100., n_cycles=12, time_bandwidth=4., **kw):
    """Extract high-gamma activity (HGA) around a central frequency.

    The HGA is extracted using the `mne.time_frequency.tfr_multitaper`
    function but the parameters provided by Julien Bastin.

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
    tf = tfr_multitaper(epoch, freq, .2 * freq, return_itc=False,
                        time_bandwidth=time_bandwidth, **kw)
    return tf


def mt_hga_split(epoch, time_bandwidth=4., hga_start=60, hga_end=160,
                 n_hga=None, norm='1/f', **kw):
    """Extract high-gamma activity (HGA) using a splited method.

    Step by step procedure :

        * Define a lineary spaced frequency vector (e.g. [60, 160]])
        * Extract each sub-gamma band
        * Normalize each sub-band
        * Take the mean across sub-bands to obtain the final HG

    Parameters
    ----------
    epoch : mne.Epochs
        Instance of mne.Epochs
    time_bandwidth : float | 4.
        Time x (Full) Bandwidth product.
    hga_start : float | 60.
        Lower boundary of the gamma band
    hga_end : float | 160.
        Higher boundary of the gamma band
    n_hga : int | None
        Number of HG sub-bands between (hga_start, hga_end)
    norm : {'1/f', 'constant'} | None
        Normalization method before taking the mean across sub-gamma bands
    kw : dict | {}
        Additional arguments are passed to the `tfr_multitaper` function.

    Returns
    -------
    tf : AverageTFR | EpochsTFR
        The averaged or single-trial HGA.
    """
    need_average = kw.get('average', False)
    if 'average' in kw.keys(): del kw['average']  # noqa
    # Define a linear frequency vector
    if not isinstance(n_hga, int):
        n_hga = int(np.round((hga_end - hga_start) / 10))
    freqs = np.linspace(hga_start, hga_end, n_hga, endpoint=True)
    n_cycles = .2 * freqs
    # Compute multitaper
    tf = tfr_multitaper(epoch, freqs, n_cycles, return_itc=False,
                        time_bandwidth=time_bandwidth, average=False,
                        **kw)
    # Get the data and apply a 1/f normalization
    tf_data = tf.data
    if norm == '1/f':
        tf_data *= freqs.reshape(1, 1, -1, 1)
    elif norm == 'constant':
        tf_data -= tf_data.mean(3, keepdims=True)
    # Mean the data across sub-gamma bands
    tf_data = tf_data.mean(2, keepdims=True)
    freqs = np.array([tf.freqs.mean()])
    # Rebuilt the EpochsTFR
    etf = EpochsTFR(tf.info, tf_data, tf.times, freqs)
    return etf.average() if need_average else etf
