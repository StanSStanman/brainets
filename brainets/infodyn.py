"""Info dynamic functions."""
import logging

import numpy as np
from xarray import DataArray
from joblib import Parallel, delayed

import mne

from brainets.gcmi import gcmi_cc, copnorm
from brainets.stats import (stat_gcmi_cluster_based, stat_gcmi_permutation)


logger = logging.getLogger('brainets')


###############################################################################
###############################################################################
#                        CHECK AND PREPARE THE DATA
###############################################################################
###############################################################################


def gcmi_prepare_data(data, dp, roi, times=None, gcrn=True, aggregate='mean',
                      modality='meg', verbose=None):
    """Prepare the M/SEEG data before computing the GCMI.

    Parameters
    ----------
    data : list
        List of length (n_subjects,). Each element of the list should either be
        an array of shape (n_trials, n_channels, n_pts), mne.Epochs,
        mne.EpochsArray, mne.EpochsTFR (i.e. non-averaged power).
        Alternatively, you can also give a xarray.DataArray instance
    dp : list
        List of arrays of shape (n_trials,) describing the behavioral variable
    roi : list
        List of arrays of shape (n_channels,) describing the ROI name of each
        channel
    times : array_like | None
        The time vector. All of the subject should have the number of time
        points. If None, a default (-1.5, 1.5) secondes vector is created. If
        MNE instances are provided, the time vector is inferred from it
    gcrn : bool | True
        Apply a Gaussian copula rank normalization.
    aggregate : {'mean', 'concat', None}
        Strategy to group sensors / channels inside an ROI. Choose either
        'mean' (e.g. mean HGA inside the roi) or 'concat' to concatenate all
        sites inside this roi
    """
    assert all([isinstance(k, (list, tuple)) for k in (data, dp)])
    assert len(data) == len(dp)
    n_suj = len(data)
    logger.info("Prepare the data of %i subjects" % n_suj)
    # check data type and convert it to numpy array
    data = [_prepare_single_subject(k, i, j, times, n) for n, (
            k, i, j) in enumerate(zip(data, dp, roi))]
    # agregate the data inside ROI
    if aggregate == 'mean':
        logger.info("    Take the mean inside each ROI")
        data = [k.groupby('roi').mean('roi') for k in data]
    elif aggregate == 'concat':
        raise NotImplementedError()
    # gaussian copula rank normalization
    if gcrn:
        logger.info("    Apply the Gaussian Copula rank normalization")
        for d in data:
            d.data = copnorm(d.data)
    return data


def _prepare_single_subject(x, dp, roi, times, n):
    """Prepare the data of a single subject."""
    # Load the MNE-instance
    if isinstance(x, str) and ('-tfr.h5' in x):     # EpochsTFR instance
        x = mne.time_frequency.read_tfrs(x)[0]
        times = x.times
    elif isinstance(x, str) and ('-epo.fif' in x):  # Epochs instance
        x = mne.read_epochs(x)
        times = x.times
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
    # time vector construction
    if times is None:
        times = np.linspace(-1.5, 1.5, data.shape[-1], endpoint=True)
    assert data.shape == (len(dp), len(roi), len(times))
    # DataArray conversion
    data = DataArray(data, coords=[dp, roi, times], name='subject%i' % n,
                     dims=['dp', 'roi', 'times'])
    return data

###############################################################################
###############################################################################
#                        COMPUTE THE GCMI
###############################################################################
###############################################################################


def gcmi_cc_mne(x, dp, smooth=None, decim=None, stat_method='cluster',
                as_single_channel=False, n_perm=1000, n_jobs=-1, verbose=None,
                **kw):
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
    decim : int | None
        Decimation factor (use it to reduce the number of time points)
    stat_method : {'cluster', 'maxstat'}
        The statistical method to use.

            * 'cluster' : perform a cluster based statistic (see
              `brainets.stats.stat_gcmi_cluster_based`)
            * 'maxstat' : perform permutations and get pvalues corrected for
              multiple comparisons using maximum statistics (see
              `brainets.stats.stat_gcmi_permutation`)
    as_single_channel : bool | False
        Reshape the data to be considered as a single channel. This can be
        usefull if all of the data are inside an ROI and you want a statistical
        correction across channels.
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
    supp : array_like | tuple
        Additional output argument that depends on the selected statistical
        mthod.

            * If `stat_method` is 'cluster', this argument represents the list
              of selected clusters for each channel (n_channels,)
            * If `stat_method` is 'maxstat', this argument represents the
              maximum permutation of each channel (n_channels,)
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

    # -------------------------------------------------------------------------
    # Consider the data as a single channel

    assert (data.ndim == 3)
    n_trials, n_channels, n_pts = data.shape
    if as_single_channel:
        logger.info("    Reshape data to be considered as a single channel")
        data = data.reshape(n_trials * n_channels, 1, n_pts)
        dp = np.tile(dp.reshape(-1, 1), (n_channels, 1)).ravel()
        n_trials, n_channels, n_pts = data.shape
    logger.info("    Data shape (%i, %i, %i)" % (n_trials, n_channels, n_pts))

    assert (data.shape[0] == len(dp)), ("The first dimension of the data (%i) "
                                        "should be equal to the length of dp "
                                        "(%i)" % (data.shape[0], len(dp)))
    assert isinstance(smooth, (type(None), int)), ("smooth should either be "
                                                   "None or an integer")
    decim = 1 if not isinstance(decim, int) else decim
    assert decim > 0, "`decim` should be an integer > 0"

    # -------------------------------------------------------------------------
    # Parallel processing can be assessed either across channels or across
    # permutations.

    is_para = n_jobs != 1
    if not is_para:                            # no parallel computing
        logger.info("    No parallel computing")
        n_jobs_chan = n_jobs_stat = 1
    elif is_para and isinstance(n_perm, int):  # parallel across permutations
        logger.info("    Parallel computing is performed across permutations")
        n_jobs_chan, n_jobs_stat = 1, n_jobs
    else:                                      # parallel across channels
        logger.info("    Parallel computing is performed across channels")
        n_jobs_chan, n_jobs_stat = n_jobs, 1

    # -------------------------------------------------------------------------
    # Get the function to compute GCMI, with or without smoothing
    fcn = _get_gcmi_smoothing(smooth, decim)
    # Get the function to compute GCMI, with or without smoothing
    fcn_stat, need_stat = _get_gcmi_stat(n_perm, stat_method, fcn, n_jobs_stat,
                                         **kw)
    if need_stat:
        logger.info("    Perform %i permutations using the %s statistical "
                    "method" % (n_perm, stat_method))

    # -------------------------------------------------------------------------
    # Compute gcmi
    out = Parallel(n_jobs=n_jobs_chan)(delayed(fcn_stat)(
        data[:, k, :], dp) for k in range(n_channels))

    # -------------------------------------------------------------------------
    # Format outputs
    if not need_stat:
        return np.asarray(out), None, None
    else:
        gcmi, pvalues, supp = zip(*out)
        gcmi, pvalues = np.stack(gcmi), np.stack(pvalues)
        supp = np.asarray(supp) if stat_method == 'maxstat' else supp
        return gcmi, pvalues, supp


###############################################################################
# Get function to compute GCMI :
#    * With or without time smoothing (_get_gcmi_smoothing)
#    * With or without statistics (_get_gcmi_stat)
###############################################################################

def _get_gcmi_smoothing(smooth, decim):
    """Get the GCMI function i.e. if a smoothing is needed or not."""
    if isinstance(smooth, int):  # smoothing needed
        logger.info("    Compute GCMI using %i time points" % (2 * smooth))

        def fcn(data, dp):  # noqa
            # dP need to be repeated
            dp = np.tile(dp.reshape(-1, 1), (1, 2 * smooth + 1)).ravel(
                order='F')
            # Compute the gcmi across time
            vec = np.arange(smooth, data.shape[1] - smooth, decim)
            gcmi = np.zeros((len(vec),), dtype=float)
            for num, k in enumerate(vec):
                _data = data[:, k - smooth:k + smooth + 1].ravel(order='F')
                gcmi[num] = gcmi_cc(_data, dp, verbose=False)
            return gcmi
    else:                        # no smoothing
        logger.info("    Compute GCMI without smoothing")

        def fcn(data, dp):  # noqa
            # Compute the gcmi across time
            vec = np.arange(0, data.shape[1], decim)
            gcmi = np.zeros((len(vec),), dtype=float)
            for num, k in enumerate(vec):
                gcmi[num] = gcmi_cc(data[:, k], dp, verbose=False)
            return gcmi
    return fcn


def _get_gcmi_stat(n_perm, stat_method, fcn, n_jobs, **kw):
    """Get the GCMI function i.e. if statistics are needed or not."""
    # Get if stats are required
    need_stat = isinstance(n_perm, int) and (n_perm > 0)
    if not need_stat:
        fcn_stat = fcn
    elif need_stat and (stat_method == 'cluster'):
        def fcn_stat(data, dp):
            return stat_gcmi_cluster_based(data, dp, fcn, n_perm=n_perm,
                                           n_jobs=n_jobs, **kw)
    elif need_stat and (stat_method == 'maxstat'):
        def fcn_stat(data, dp):
            return stat_gcmi_permutation(data, dp, fcn, n_perm=n_perm,
                                         n_jobs=n_jobs)
    return fcn_stat, need_stat


if __name__ == '__main__':

    def generate_data(n_trials, n_channels, n_pts, n_roi=3):
        x = np.random.rand(n_trials, n_channels, n_pts)
        dp = np.random.rand(n_trials)
        roi = np.random.randint(0, n_roi, n_channels)
        times = np.linspace(-1.4, 1.4, n_pts, endpoint=True)
        return x, dp, roi, times
    # -------------------------------------------------------------------------
    # Dataset 1
    x_1, dp_1, roi_1, times = generate_data(50, 20, 100)
    # Dataset 2
    x_2, dp_2, roi_2, times = generate_data(47, 37, 100)
    # Concatenate datasets
    x = [x_1, x_2]
    dp = [dp_1, dp_2]
    roi = [roi_1, roi_2]
    data = gcmi_prepare_data(x, dp, roi, times=times)
    # -------------------------------------------------------------------------

    # import matplotlib.pyplot as plt

    # ###########################################################################
    # n_trials = 30
    # n_channels = 1
    # n_perm = 30
    # stat_method = 'maxstat'
    # smooth = 5
    # decim = 1
    # threshold = 25
    # as_single_channel = False
    # n_jobs = -1
    # ###########################################################################

    # np.random.seed(120)
    # x = np.arange(n_trials)
    # y = np.random.rand(n_trials, n_channels, 1000)
    # y[..., 50:100] *= x.reshape(-1, 1, 1)
    # y[..., 200:400] *= x.reshape(-1, 1, 1)

    # kw = dict(stat_method=stat_method, n_perm=n_perm, smooth=smooth,
    #           threshold=threshold, n_jobs=n_jobs, decim=decim,
    #           as_single_channel=as_single_channel)

    # pval = np.ones((y.shape[2],))
    # if not isinstance(n_perm, int):
    #     gcmi, _, _ = gcmi_cc_mne(y, x, **kw)
    # elif isinstance(n_perm, int) and (stat_method == 'cluster'):
    #     gcmi, pvalues, cluster = gcmi_cc_mne(y, x, **kw)
    #     pval = pvalues.mean(0)
    # elif isinstance(n_perm, int) and (stat_method == 'maxstat'):
    #     gcmi, pvalues, p_max = gcmi_cc_mne(y, x, **kw)
    #     pval = pvalues.mean(0)

    # plt.subplot(211)
    # plt.plot(gcmi.mean(0))
    # plt.subplot(212)
    # plt.plot(pval)
    # plt.show()
