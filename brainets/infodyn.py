"""Info dynamic functions."""
import logging

import numpy as np
from joblib import Parallel, delayed

import mne

from .gcmi import gcmi_cc


logger = logging.getLogger('brainets')


def gcmi_cc_mne(x, dp, smooth=None, n_jobs=-1, verbose=None):
    """Compute the Gaussian-Copula Mutual Information on MNE instance.

    Parameters
    ----------
    x : mne.Epochs | mne.EpochsTFR | str
        The data to use. Should either be :

            * array of shape (n_trials, n_channels, n_pts)
            * mne.Epochs
            * mne.EpochsTFR (i.e. non-averaged power)
            * Path (string) to an epoch file (-epo.fif) or TFR file (tfr.h5)
    dp : array_like
        Contingency vector of shape (n_trials,)
    smooth : int | None
        Time smoothing factor
    n_jobs : int | -1
        Number of jobs to use for parallel computing (use -1 to use all jobs)

    Returns
    -------
    gcmi : array_like
        The gcmi array of shape (n_channels, n_pts) where n_pts is the number
        of time points ()

    Example
    -------
    >>> Run gcmi between high-gamma power and dP
    >>> from brainets.spectral import mt_hga
    >>> tf = mt_hga(...)
    >>> gcmi_cc_mne(tf, dp)
    """
    dp = np.asarray(dp)
    assert dp.ndim == 1, "dp should be a row vector"
    # Load the MNE-instance
    if isinstance(x, str) and ('-tfr.h5' in x):     # EpochsTFR instance
        x = mne.time_frequency.read_tfrs(x)[0]
    elif isinstance(x, str) and ('-epo.fif' in x):  # Epochs instance
        x = mne.read_epochs(x)
    # Check inputs
    if isinstance(x, (mne.Epochs, mne.EpochsArray,
                      mne.time_frequency.EpochsTFR)):
        data = x.data
    elif isinstance(x, np.ndarray):
        data = x
    else:
        raise TypeError("x of type %s not supported" % str(type(x)))
    # Handle multi-dimentional arrays
    if data.ndim == 4:  # TF : (n_trials, n_channels, n_freqs, n_pts)
        if data.shape[2] == 1:
            data = data[..., 0, :]
        else:
            data = data.mean(2)
            logger.warning("Multiple frequencies detected. Take the mean "
                           "across frequencies")
    assert (data.ndim == 3)
    n_trials, n_channels, n_pts = data.shape
    assert (data.shape[0] == len(dp)), ("The first dimension of the data (%i) "
                                        "should be equal to the length of dp "
                                        "(%i)" % (data.shape[0], len(dp)))
    assert isinstance(smooth, (type(None), int)), ("smooth should either be "
                                                   "None or an integer")
    # Compute gcmi
    gcmi = Parallel(n_jobs=n_jobs)(delayed(_gcmi)(data[:, k, :], dp, smooth)
                                   for k in range(n_channels))
    return np.asarray(gcmi)


def _gcmi(data, dp, smooth):
    """Function to run in parralel."""
    if isinstance(smooth, int):
        vec = np.arange(smooth, data.shape[1] - smooth)
        dp = np.tile(dp.reshape(-1, 1), (1, 2 * smooth)).ravel(order='F')
    else:
        vec = np.arange(data.shape[1])
    gcmi = np.zeros((len(vec),), dtype=float)
    for num, k in enumerate(vec):
        if isinstance(smooth, int):
            _data = data[:, k - smooth:k + smooth].ravel(order='F')
        else:
            _data = data[:, k]
        gcmi[num] = gcmi_cc(_data, dp)
    return gcmi
