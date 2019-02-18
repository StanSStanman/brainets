"""Statistical functions."""
import numpy as np


def cluster_detection(raw, th, min_cluster_size=None):
    """Detect clusters in a raw vector.

    Parameters
    ----------
    raw : array_like
        1D array in which clusters need to be detected
    th : float
        The threshold
    min_cluster_size : int | None
        Minimum cluster size to detect

    Returns
    -------
    clusters : list
        List of slice objects indicating where cluster are located
    """
    min_cl = -1 if not isinstance(min_cluster_size, int) else min_cluster_size
    # Transient detection
    _transient = np.diff(np.r_[0, (raw > th).astype(int), 0])
    # Build the (start, end)
    _start = np.where(_transient == 1.)[0]
    _end = np.where(_transient == -1.)[0]
    assert len(_start) == len(_end)
    return [slice(s, e) for s, e in zip(_start, _end) if e - s >= min_cl]


def stat_gcmi_cluster_based(raw, dp, fcn, n_perm=1000, threshold=1):
    """Perform cluster based statistics.

    This function perform a cluster based statistic test between the data
    and a second continous variable.

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
    threshold : float | 1.
        Threshold for cluster detection according to a percentile of data
        repartition (by default 1%)

    Returns
    -------
    gcmi : array_like
        True GCMI estimations of shape (n_pts,)
    pvalues : array_like
        P-values  array of shape (n_pts,)
    """
    assert raw.ndim == 2, "Only works for two-dimentional arrays"
    assert raw.shape[0] == len(dp), "`raw` shape should be (n_trials, n_pts)"
    # Get non modified gcmi
    gcmi = fcn(raw, dp)
    # Cluster detection
    _th = np.percentile(gcmi, 100. - threshold)
    clusters = cluster_detection(gcmi, _th)
    # Get the GCMI sum inside clusters
    gcmi_cl = np.array([gcmi[k].sum() for k in clusters])
    # Perform permutations
    perm_clusters = np.zeros((n_perm, len(clusters)), dtype=float)
    for k in range(n_perm):
        dp_perm = dp.copy()
        np.random.shuffle(dp_perm)
        gcmi_perm = fcn(raw, dp_perm)
        perm_clusters[k, :] = np.array([gcmi_perm[k].sum() for k in clusters])
    # Get associated p-values
    pval_cl = (gcmi_cl.reshape(1, -1) < perm_clusters).sum(0) / n_perm
    pvalues = np.ones((len(gcmi),), dtype=float)
    for c, p in zip(clusters, pval_cl):
        pvalues[c] = p
    return gcmi, pvalues, clusters


def stat_gcmi_permutation(raw, dp, fcn, n_perm=1000):
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

    Returns
    -------
    gcmi : array_like
        True GCMI estimations of shape (n_pts,)
    pvalues : array_like
        Corrected p-values across the time-dimension of shape (n_pts,)
    """
    assert raw.ndim == 2, "Only works for two-dimentional arrays"
    assert raw.shape[0] == len(dp), "`raw` shape should be (n_trials, n_pts)"
    # Get non modified gcmi
    gcmi = fcn(raw, dp)
    # Perform permutations
    perm = np.zeros((n_perm, len(gcmi)), dtype=float)
    for k in range(n_perm):
        dp_perm = dp.copy()
        np.random.shuffle(dp_perm)
        perm[k, :] = fcn(raw, dp_perm)
    # Get corrected p-values
    pvalues = np.ones((len(gcmi),), dtype='float')
    p_max = perm.max()
    pvalues[gcmi > p_max] = 1. / n_perm
    return gcmi, pvalues, p_max
