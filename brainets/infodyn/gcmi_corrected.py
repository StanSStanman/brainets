"""Info dynamic functions."""
import logging

import numpy as np
import pandas as pd
from xarray import DataArray

from brainets.gcmi import nd_gccmi_ccd
from brainets.stats import stat_gcmi_cluster_based, stat_gcmi_permutation


logger = logging.getLogger('brainets')


def gcmi_corrected(x, n_perm=1000, stat='cluster', n_jobs=-1,
                   as_dataframe=False, verbose=None, **kw):
    """Compute the Gaussian-Copula Mutual Information.

    This function computes the GCMI across subjects, roi and time. It also
    evaluate statistics.

    Parameters
    ----------
    x : list
        List of prepared arrays. See
        :func:`brainets.infodyn.gcmi_prepare_data`
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
    need_stat = isinstance(n_perm, int) and (n_perm > 0)

    # Get the function to compute GCMI
    fcn = _get_gcmi_fcn()

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


def _get_gcmi_fcn():
    """Get the GCMI function."""
    def fcn(x):  # noqa
        # Compute the gcmi across time
        gcmi = np.zeros((len(x), x[0].shape[0]), dtype=float)
        for nr, r in enumerate(x):
            data, dp, z = r.data, r.dp.values, r.attrs['z']
            dp = np.tile(dp, (data.shape[0], 1, 1))
            gcmi[nr, :] = nd_gccmi_ccd(data, dp, z, shape_checking=False,
                                       gcrn=False)
        return gcmi
    return fcn
