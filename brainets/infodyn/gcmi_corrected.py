"""Info dynamic functions."""
import logging

import numpy as np
import pandas as pd
from xarray import DataArray

from brainets.gcmi import gccmi_ccd
from brainets.stats import stat_gcmi_cluster_based, stat_gcmi_permutation


logger = logging.getLogger('brainets')


def gcmi_corrected(x, smooth=None, decim=None, n_perm=1000, stat='cluster',
                   n_jobs=-1, as_dataframe=False, verbose=None, **kw):
    """Compute the Gaussian-Copula Mutual Information.

    This function computes the GCMI across subjects, roi and time. It also
    evaluate statistics.

    Parameters
    ----------
    x : list
        List of prepared arrays. See
        :func:`brainets.infodyn.gcmi_prepare_data`
    smooth : int | None
        Time smoothing factor
    decim : int | None
        Decimation factor (use it to reduce the number of time points)
    n_perm : int | 1000
        Number of permutations to perform
    stat : {'cluster', 'maxstat'}
        The statistical method to use for the correction. Use either 'cluster'
        for a cluster-based approach or 'maxstat' for using maximum statistics
    n_jobs : int | -1
        Number of jobs to use for parallel computing (use -1 to use all jobs)
    as_dataframe : bool | False
        Return results as an ROI organized DataFrame
    kw : dict | {}
        Additional input arguments to pass to the selected statistic method
        (see :func:`brainets.stats.stat_gcmi_cluster_based` and
        :func:`brainets.stats.stat_gcmi_permutation`)

    Returns
    -------
    gcmi : array_like
        The gcmi array of shape (n_roi, n_pts) where n_pts is the number
        of time points. Output type depends on the `as_dataframe` input
    pvalues : array_like | tuple
        p-values array of shape (n_roi, n_pts,). Output type depends on the
        `as_dataframe` input

    See also
    --------
    brainets.infodyn.gcmi_prepare_data
    brainets.stats.stat_gcmi_cluster_based
    brainets.stats.stat_gcmi_permutation
    """
    # Inputs checking
    assert isinstance(x, list) and all([isinstance(k, DataArray) for k in x])
    assert isinstance(smooth, (type(None), int)), ("smooth should either be "
                                                   "None or an integer")
    decim = 1 if not isinstance(decim, int) else decim
    assert decim > 0, "`decim` should be an integer > 0"
    need_stat = isinstance(n_perm, int) and (n_perm > 0)

    # Get the function to compute GCMI, with or without smoothing
    fcn = _get_gcmi_smoothing(smooth, decim)

    # Compute corrected or not GCMI
    if not need_stat:
        logger.info("Compute GCMI without statistics")
        gcmi, pvalues = fcn(x), None
    elif stat == 'cluster':
        logger.info("Compute GCMI using a cluster-based approach")
        gcmi, pvalues = stat_gcmi_cluster_based(x, fcn, n_perm=n_perm,
                                                n_jobs=n_jobs, **kw)
    elif stat == 'maxstat':
        logger.info("Compute GCMI using maximum statistics")
        gcmi, pvalues = stat_gcmi_permutation(x, fcn, n_perm=n_perm,
                                              n_jobs=n_jobs)

    # Pandas formatting
    if as_dataframe:
        roi = [k.name for k in x]
        times = np.linspace(x[0].times[0], x[0].times[-1], gcmi.shape[1],
                            endpoint=True)
        gcmi = pd.DataFrame(gcmi.T, index=times, columns=roi)
        if isinstance(pvalues, (np.ndarray, np.ma.core.MaskedArray)):
            pvalues = pd.DataFrame(pvalues.T, index=times, columns=roi)

    return gcmi, pvalues


def _get_gcmi_smoothing(smooth, decim):
    """Get the GCMI function i.e. if a smoothing is needed or not."""
    if isinstance(smooth, int):  # smoothing needed
        logger.info("    Compute GCMI using %i time points" % (2 * smooth))

        def fcn(x):  # noqa
            # dP need to be repeated
            # Compute the gcmi across time
            vec = np.arange(smooth, x[0].shape[1] - smooth, decim)
            gcmi = np.zeros((len(x), len(vec),), dtype=float)
            for nr, r in enumerate(x):
                data, dp, z = r.data, r.dp.values, r.attrs['z']
                zm = int(np.max(z) + 1)
                dp = np.tile(dp.reshape(-1, 1), (1, 2 * smooth + 1)).ravel(
                    order='F')
                z = np.tile(z.reshape(-1, 1), (1, 2 * smooth + 1)).ravel(
                    order='F')
                for nt, k in enumerate(vec):
                    _data = data[:, k - smooth:k + smooth + 1].ravel(order='F')
                    gcmi[nr, nt] = gccmi_ccd(_data, dp, z, zm,
                                             verbose=False)[0]
            return gcmi
    else:                        # no smoothing
        logger.info("    Compute GCMI without smoothing")

        def fcn(x):  # noqa
            # Compute the gcmi across time
            vec = np.arange(0, x[0].shape[1], decim)
            gcmi = np.zeros((len(x), len(vec),), dtype=float)
            for nr, r in enumerate(x):
                data, dp, z = r.data, r.dp.values, r.attrs['z']
                zm = int(np.max(z) + 1)
                for nt, k in enumerate(vec):
                    gcmi[nr, nt] = gccmi_ccd(data[:, k], dp, z, zm,
                                             verbose=False)[0]
            return gcmi
    return fcn


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from brainets.infodyn import gcmi_prepare_data

    def generate_data(n_trials, n_channels, n_pts, n_roi=3):  # noqa
        x = np.random.rand(n_trials, n_channels, n_pts)
        # dp = np.arange(n_trials)
        dp = np.random.rand(n_trials)
        x[..., 30:70] *= dp.reshape(-1, 1, 1)
        x[..., 120:150] *= dp.reshape(-1, 1, 1)
        roi = ['roi%i' % k for k in np.random.randint(0, n_roi, n_channels)]
        times = np.linspace(-1.4, 1.4, n_pts, endpoint=True)
        return x, dp, roi, times

    # -------------------------------------------------------------------------
    # Dataset 1
    x_1, dp_1, roi_1, times = generate_data(50, 20, 200)
    # Dataset 2
    x_2, dp_2, roi_2, times = generate_data(47, 37, 200, 3)
    # Concatenate datasets
    x = [x_1, x_2]
    dp = [dp_1, dp_2]
    roi = [roi_1, roi_2]
    x = gcmi_prepare_data(x, dp, roi, times=times, aggregate='mean')

    gcmi, pvalues = gcmi_corrected(x, smooth=5, n_perm=30, alpha=0.05,
                                   correction='bonferroni', stat='cluster',
                                   reduce='max', as_dataframe=False)

    plt.subplot(121)
    plt.pcolormesh(gcmi)
    plt.subplot(122)
    plt.pcolormesh(pvalues)
    plt.show()
