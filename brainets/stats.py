"""Statistical functions."""
import numpy as np

from joblib import Parallel, delayed


def stat_gcmi_cluster_based(raw, dp, fcn, n_perm=1000, threshold=1, n_jobs=-1):
    """Perform cluster based statistics.

    This function perform a cluster based statistic test between the data
    and a second continous variable.

    Parameters
    ----------
    raw : array_like
        The data (e.g. HGA) of shape (n_trials, n_roi, n_pts)
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
    n_jobs : int | -1
        Number of jobs to use for parallel computing (use -1 to use all jobs)

    Returns
    -------
    gcmi : array_like
        True GCMI estimations of shape (n_pts,)
    pvalues : array_like
        P-values  array of shape (n_pts,)
    """
    assert (raw.ndim == 3) and (raw.shape[0] == len(dp))

    # True GCMI estimation
    gcmi = fcn(raw, dp)  # (n_roi, n_pts)

    # Compute permutations
    perm = Parallel(n_jobs=n_jobs)(delayed(_compute_gcmi_permutations)(
        raw, dp, fcn) for k in range(n_perm))
    perm = np.asarray(perm)  # (n_perm, n_roi, n_pts)

    # Infer statistical threshold
    th_pval = np.sum(perm < gcmi, axis=0) / n_perm
    print(th_pval)
    0/0

    ###########################################################################
    # Cluster detection and reduction


    # Get non modified gcmi
    # Cluster detection
    _th = np.percentile(gcmi, 100. - threshold)
    clusters = cluster_detection(gcmi, _th)
    # Get the GCMI sum inside clusters
    gcmi_cl = np.array([gcmi[k].sum() for k in clusters])
    # Perform permutations
    perm_clusters = Parallel(n_jobs=n_jobs)(delayed(_para_gcmi_cluster)(
        dp, raw, clusters, fcn) for k in range(n_perm))
    perm_clusters = np.array(perm_clusters)
    # Get associated p-values
    pval_cl = (gcmi_cl.reshape(1, -1) < perm_clusters).sum(0) / n_perm
    pvalues = np.ones((len(gcmi),), dtype=float)
    for c, p in zip(clusters, pval_cl):
        pvalues[c] = p
    return gcmi, pvalues, clusters


def _compute_gcmi_permutations(raw, dp, fcn):
    """Compute GCMI between the data and the shuffle version of dp."""
    dp_perm = dp.copy()
    np.random.shuffle(dp_perm)
    return fcn(raw, dp_perm)



def cluster_reduction(gcmi, th, reduce='sum'):
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

    return gcmi_cl, clusters



def _para_gcmi_cluster(dp, raw, clusters, fcn):
    """Parallel function to be runned for permutations."""
    dp_perm = dp.copy()
    np.random.shuffle(dp_perm)
    gcmi_perm = fcn(raw, dp_perm)
    return np.array([gcmi_perm[k].sum() for k in clusters])


def stat_gcmi_permutation(raw, dp, fcn, n_perm=1000, n_jobs=-1):
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
        True GCMI estimations of shape (n_pts,)
    pvalues : array_like
        Corrected p-values across the time-dimension of shape (n_pts,)
    """
    assert raw.ndim == 2, "Only works for two-dimentional arrays"
    assert raw.shape[0] == len(dp), "`raw` shape should be (n_trials, n_pts)"
    # Get non modified gcmi
    gcmi = fcn(raw, dp)
    # Perform permutations
    perm = Parallel(n_jobs=n_jobs)(delayed(_para_gcmi_maxstat)(
        dp, raw, fcn) for k in range(n_perm))
    perm = np.array(perm)
    # Get corrected p-values
    pvalues = np.ones((len(gcmi),), dtype='float')
    p_max = perm.max()
    pvalues[gcmi > p_max] = 1. / n_perm
    return gcmi, pvalues, p_max


def _para_gcmi_maxstat(dp, raw, fcn):
    """Parallel function to be runned for permutations."""
    dp_perm = dp.copy()
    np.random.shuffle(dp_perm)
    return fcn(raw, dp_perm)


if __name__ == '__main__':
    x = np.random.rand(80, 10, 100)
    dp = np.random.rand(80)

    def fcn(x, dp): return np.sum(x, axis=0) * dp[17]

    stat_gcmi_cluster_based(x, dp, fcn, n_perm=20)