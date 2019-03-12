"""Statistical functions."""
import numpy as np

from mne.stats import fdr_correction, bonferroni_correction

from joblib import Parallel, delayed


def stat_gcmi_cluster_based(x, fcn, n_perm=1000, correction='fdr', alpha=.05,
                            reduce='sum', n_jobs=-1):
    """Perform cluster based statistics.

    This function performs the following steps :

        * Compute the true GCMI
        * Compute the permutations and infer the threshold
        * Find and reduce the clusters
        * Detect the clusters on the permutations
        * Compare the cluster size between the true GCMI and the permutations

    Parameters
    ----------
    x : list
        List of data per subject.
    fcn : function
        The function to use to evaluate the information shared. This function
        should accept two inputs :

            * raw of shape (n_trials, n_pts)
            * dp of shape (n_trials,)

        At the end, it should returns a single vector of GCMI computed across
        trials of shape (n_pts,)
    n_perm : int | 1000
        Number of permutations to perform
    correction : {'fdr', 'bonferroni', 'maxstat'}
        Correction type to apply to the p-values for the inference of the
        threshold to use to detect the clusters.
    alpha : float | .05
        Error rate to use if correction is 'fdr' or 'bonferroni'
    reduce : {'sum', 'length', 'max'}
        The function to reduce GCMI values inside the cluster. Use either :

            * 'sum' : cluster-mass
            * 'max' : cluster-height
            * 'length' : cluster-extent
    n_jobs : int | -1
        Number of jobs to use for parallel computing (use -1 to use all jobs)

    Returns
    -------
    gcmi : array_like
        True GCMI estimations of shape (n_roi, n_pts)
    pvalues : array_like
        Corrected p-values of shape (n_roi, n_pts)
    """
    # True GCMI estimation
    gcmi = fcn(x)  # (n_roi, n_pts)
    # Compute permutations
    perm = Parallel(n_jobs=n_jobs)(delayed(_gccmi_permutations)(
        x, fcn) for k in range(n_perm))
    perm = np.asarray(perm)  # (n_perm, n_roi, n_pts)
    # Infer statistical threshold
    if correction in ['fdr', 'bonferroni']:
        fcs = fdr_correction if correction == 'fdr' else bonferroni_correction
        th_pval = np.sum(gcmi < perm, axis=0) / n_perm
        th = gcmi[~fcs(th_pval, alpha)[0]].max()
    elif correction == 'maxstat':
        th = np.percentile(perm.max(axis=(1, 2)), 100. * (1. - alpha))
    # Cluster detection and reduction
    gcmi_cl, clusters = cluster_reduction(gcmi, th, reduce=reduce)
    # Find clusters on permutations
    perm_cl = []
    for p in range(n_perm):
        perm_cl += [cluster_reduction(perm[p, ...], th, reduce=reduce,
                                      maximum=True)[0]]
    perm_cl = np.asarray(perm_cl).max(1)  # (n_perm,)
    # Test cluster size significance
    pval_cl = [[(i < perm_cl).sum() / n_perm for i in k] for k in gcmi_cl]
    # Reformat p-values
    pvalues, mask = np.ones_like(gcmi), np.ones_like(gcmi, dtype=bool)
    for num, (rp, rc) in enumerate(zip(pval_cl, clusters)):
        for p, c in zip(rp, rc):
            pvalues[num, c] = p
            mask[num, c] = False
    pvalues = np.ma.masked_array(pvalues, mask=mask)
    return gcmi, pvalues


def _gccmi_permutations(x, fcn):
    """Compute GCMI between the data and the shuffle version of dp.

    Note that this function compute inner permutations i.e. randomly permutes
    the dp per subject
    """
    x_perm = []
    for r in x:
        dp, z = r.dp.values, r.attrs['z']
        for u_z in np.unique(z):
            is_z = z == u_z
            dp[is_z] = np.random.permutation(dp[is_z])
        x_perm += [r.assign_coords(dp=dp)]
    return fcn(x_perm)


def cluster_reduction(gcmi, th, reduce='sum', maximum=False):
    """Detect and reduce clusters.

    The following steps are performed :

        * Detect where the data exceed the threshold
        * Detect clusters inside each roi
        * Reduce each detected cluster

    Parameters
    ----------
    gcmi : array_like
        GCMI 2D array of shape (n_roi, n_times) in which clusters need to be
        detected and reduced
    th : float
        The threshold
    reduce : {'sum', 'length', 'max'}
        The function to reduce GCMI values inside the cluster. Use either :

            * 'sum' : cluster-mass
            * 'max' : cluster-height
            * 'length' : cluster-extent
    maximum : bool | False
        Get only clusters with maximum size

    Returns
    -------
    gcmi_cl : list
        List of length n_roi. Each element of the list contains the reduced
        gcmi inside each cluster. This is applied across the time dimension
    clusters : list
        List of length n_roi containing the detected clusters
    """
    # Reducing function
    fcn = dict(sum=np.sum, max=np.max, length=len)[reduce]

    # Transient detection
    is_over = (gcmi > th).astype(int)
    pad = np.zeros((gcmi.shape[0], 1), dtype=float)
    transients = np.diff(np.c_[pad, is_over, pad], axis=1)

    # Get values inside clusters
    gcmi_cl, clusters = [], []
    for r, tr in zip(gcmi, transients):
        start, end = np.where(tr == 1.)[0], np.where(tr == -1.)[0]
        assert len(start) == len(end)
        cl = [slice(s, e) for s, e in zip(start, end)]
        clusters += [cl]
        gcmi_cl += [[fcn(r[c]) for c in cl]]

    # Max size
    if maximum:
        gcmi_cl = [np.max(k) if len(k) else 0. for k in gcmi_cl]

    return gcmi_cl, clusters


def stat_gcmi_permutation(x, fcn, n_perm=1000, n_jobs=-1):
    """Perform GCMI permutations and correct for multiple comparisons.

    Parameters
    ----------
    raw : array_like
        The data of shape (n_trials, n_pts)
    dp : array_like
        The contingency variable
    fcn : function
        The function to use to evaluate the information shared. This function
        should accept two inputs :

            * raw of shape (n_trials, n_pts)
            * dp of shape (n_trials,)

        At the end, it should returns a single vector of GCMI computed across
        trials of shape (n_pts,)
    n_perm : int | 1000
        Number of permutations to perform
    n_jobs : int | -1
        Number of jobs to use for parallel computing (use -1 to use all jobs)

    Returns
    -------
    gcmi : array_like
        True GCMI estimations of shape (n_roi, n_pts)
    pvalues : array_like
        Corrected p-values of shape (n_roi, n_pts)
    """
    # True GCMI estimation
    gcmi = fcn(x)  # (n_roi, n_pts)
    # Compute permutations
    perm = Parallel(n_jobs=n_jobs)(delayed(_gccmi_permutations)(
        x, fcn) for k in range(n_perm))
    perm = np.asarray(perm)  # (n_perm, n_roi, n_pts)
    # Get corrected p-values
    pvalues = np.ones_like(gcmi, dtype='float')
    p_max = perm.max()
    pvalues[gcmi > p_max] = 1. / n_perm
    return gcmi, pvalues
