"""Info dynamic functions."""
import logging

import numpy as np
from joblib import Parallel, delayed

import mne

from brainets.gcmi import gcmi_cc
from brainets.stats import (stat_gcmi_cluster_based, stat_gcmi_permutation)


logger = logging.getLogger('brainets')


def gcmi_cc_mne(x, dp, smooth=None, stat_method='cluster', n_perm=1000,
                n_jobs=-1, verbose=None, **kw):
    """Compute the Gaussian-Copula Mutual Information on MNE instance.

    Parameters
    ----------
    x : mne.Epochs | mne.EpochsTFR | str
        The data to use. Should either be :

            * array of shape (n_trials, n_channels, n_pts)
            * mne.Epochs or mne.EpochsArray
            * mne.EpochsTFR (i.e. non-averaged power)
            * Path (string) to an epoch file (-epo.fif) or TFR file (tfr.h5)
    dp : array_like
        Contingency vector of shape (n_trials,)
    smooth : int | None
        Time smoothing factor
    stat_method : {'cluster', 'maxstat'}
        The statistical method to use.

            * 'cluster' : perform a cluster based statistic (see
              `brainets.stats.stat_gcmi_cluster_based`)
            * 'maxstat' : perform permutations and get pvalues corrected for
              multiple comparisons using maximum statistics (see
              `brainets.stats.stat_gcmi_permutation`)
    n_perm : int | 1000
        Number of permutations to perform
    n_jobs : int | -1
        Number of jobs to use for parallel computing (use -1 to use all jobs)
    kw : dict | {}
        Additional input arguments to pass to the selected statistic method

    Returns
    -------
    gcmi : array_like
        The gcmi array of shape (n_channels, n_pts) where n_pts is the number
        of time points. If not permutations are performed, this is the only
        returned argument
    pvalues : array_like | tuple
        p-values array of shape (n_channels, n_pts,)
    clusters : tuple
        The selected clusters (only if `stat_method` is 'cluster')

    Example
    -------
    >>> Run gcmi between high-gamma power and dP
    >>> from brainets.spectral import mt_hga
    >>> tf = mt_hga(...)
    >>> gcmi_cc_mne(tf, dp)
    """
    # -------------------------------------------------------------------------
    # Inputs checking
    dp = np.asarray(dp)
    assert dp.ndim == 1, "dp should be a row vector"
    assert stat_method in ['cluster', 'maxstat']
    # Load the MNE-instance
    if isinstance(x, str) and ('-tfr.h5' in x):     # EpochsTFR instance
        x = mne.time_frequency.read_tfrs(x)[0]
    elif isinstance(x, str) and ('-epo.fif' in x):  # Epochs instance
        x = mne.read_epochs(x)
    # Check inputs
    if isinstance(x, (mne.Epochs, mne.EpochsArray,
                      mne.time_frequency.EpochsTFR)):
        data = x.get_data()
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

    # -------------------------------------------------------------------------
    # Get the function to compute GCMI, with or without smoothing
    fcn = _get_gcmi_smoothing(smooth)
    # Get the function to compute GCMI, with or without smoothing
    fcn_stat, need_stat = _get_gcmi_stat(n_perm, stat_method, fcn, **kw)

    # -------------------------------------------------------------------------
    # Compute gcmi
    out = Parallel(n_jobs=n_jobs)(delayed(fcn_stat)(
        data[:, k, :], dp) for k in range(n_channels))

    # -------------------------------------------------------------------------
    # Format outputs
    if not need_stat:
        return np.asarray(out), None, None
    else:
        if stat_method == 'cluster':
            gcmi, pvalues, clusters = zip(*out)
            gcmi, pvalues = np.stack(gcmi), np.stack(pvalues)
            return gcmi, pvalues, clusters
        elif stat_method == 'maxstat':
            gcmi, pvalues = zip(*out)
            gcmi, pvalues = np.stack(gcmi), np.stack(pvalues)
            return gcmi, pvalues, None


###############################################################################
# Get function to compute GCMI :
#    * With or without time smoothing (_get_gcmi_smoothing)
#    * With or without statistics (_get_gcmi_stat)
###############################################################################

def _get_gcmi_smoothing(smooth):
    """Get the GCMI function i.e. if a smoothing is needed or not."""
    if isinstance(smooth, int):  # smoothing needed
        logger.info("    Compute GCMI using %i time points" % (2 * smooth))
        def fcn(data, dp):  # noqa
            # dP need to be repeated
            dp = np.tile(dp.reshape(-1, 1), (1, 2 * smooth + 1)).ravel(
                order='F')
            # Compute the gcmi across time
            vec = np.arange(smooth, data.shape[1] - smooth)
            gcmi = np.zeros((len(vec),), dtype=float)
            for num, k in enumerate(vec):
                _data = data[:, k - smooth:k + smooth + 1].ravel(order='F')
                gcmi[num] = gcmi_cc(_data, dp)
            return gcmi
    else:                        # no smoothing
        logger.info("    Compute GCMI without smoothing")
        def fcn(data, dp):  # noqa
            # Compute the gcmi across time
            vec = np.arange(data.shape[1])
            gcmi = np.zeros((len(vec),), dtype=float)
            for num, k in enumerate(vec):
                gcmi[num] = gcmi_cc(data[:, k], dp)
            return gcmi
    return fcn


def _get_gcmi_stat(n_perm, stat_method, fcn, **kw):
    """Get the GCMI function i.e. if statistics are needed or not."""
    # Get if stats are required
    need_stat = isinstance(n_perm, int) and (n_perm > 0)
    if not need_stat:
        fcn_stat = fcn
    elif need_stat and (stat_method == 'cluster'):
        def fcn_stat(data, dp):
            return stat_gcmi_cluster_based(data, dp, fcn, n_perm=n_perm, **kw)
    elif need_stat and (stat_method == 'maxstat'):
        def fcn_stat(data, dp):
            return stat_gcmi_permutation(data, dp, fcn, n_perm=n_perm)
    return fcn_stat, need_stat


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ###########################################################################
    n_trials = 30
    n_channels = 1
    n_perm = 10
    stat_method = 'cluster'
    smooth = 5
    threshold = 25
    n_jobs = -1
    ###########################################################################

    np.random.seed(120)
    x = np.arange(n_trials)
    y = np.random.rand(n_trials, n_channels, 1000)
    y[..., 50:100] *= x.reshape(-1, 1, 1)
    y[..., 200:400] *= x.reshape(-1, 1, 1)

    kw = dict(stat_method=stat_method, n_perm=n_perm, smooth=smooth,
              threshold=threshold, n_jobs=n_jobs)

    pval = np.ones((y.shape[2],))
    if not isinstance(n_perm, int):
        gcmi = gcmi_cc_mne(y, x, **kw)
    elif isinstance(n_perm, int) and (stat_method == 'cluster'):
        gcmi, pvalues, cluster = gcmi_cc_mne(y, x, **kw)
        pval = pvalues.mean(0)
    elif isinstance(n_perm, int) and (stat_method == 'maxstat'):
        gcmi, pvalues = gcmi_cc_mne(y, x, **kw)
        pval = pvalues.mean(0)

    plt.subplot(211)
    plt.plot(gcmi.mean(0))
    plt.subplot(212)
    plt.plot(pval)
    plt.show()
