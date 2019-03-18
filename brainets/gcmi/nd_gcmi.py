"""Multi-dimentional Gaussian copula mutual information estimation.

| **Authors** : Robin AA. Ince
| **Original code** : https://github.com/robince/gcmi
| **Reference** :
| RAA Ince, BL Giordano, C Kayser, GA Rousselet, J Gross and PG Schyns "A statistical framework for neuroimaging data analysis based on mutual information estimated via a Gaussian copula" Human Brain Mapping (2017) 38 p. 1541-1573 doi:10.1002/hbm.23471

| **Multi-dimentional adaptation**
| **Authors** : Etienne Combrisson
| **Contact** : e.combrisson@gmail.com
"""
import numpy as np
from scipy.special import psi, ndtri


def ctransform(x):
    """Copula transformation (empirical CDF).

    Parameters
    ----------
    x : array_like
        Array of data. The trial axis should be the last one

    Returns
    -------
    xr : array_like
        Empirical CDF value along the last axis of x. Data is ranked and scaled
        within [0 1] (open interval)
    """
    xi = np.argsort(x)
    xr = np.argsort(xi).astype(float)
    xr += 1.
    xr /= float(xr.shape[-1] + 1)
    return xr


def copnorm(x):
    """Copula normalization.

    Parameters
    ----------
    x : array_like
        Array of data. The trial axis should be the last one

    Returns
    -------
    cx : array_like
        Standard normal samples with the same empirical CDF value as the input.
        Operates along the last axis
    """
    cx = ndtri(ctransform(x))
    # cx = sp.stats.norm.ppf(transform)
    return cx


def nd_reshape(x, mvaxis=None, traxis=-1):
    """Multi-dimentional reshaping.

    This function is used to be sure that an nd array has a correct shape
    of (..., mvaxis, traxis).

    Parameters
    ----------
    x : array_like
        Multi-dimentional array
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered

    Returns
    -------
    x_rsh : array_like
        The reshaped multi-dimentional array of shape (..., mvaxis, traxis)
    """
    assert isinstance(traxis, int)
    traxis = np.arange(x.ndim)[traxis]

    # Create an empty mvaxis axis
    if not isinstance(mvaxis, int):
        x = x[..., np.newaxis]
        mvaxis = -1
    assert isinstance(mvaxis, int)
    mvaxis = np.arange(x.ndim)[mvaxis]

    # move the multi-variate and trial axis
    x = np.moveaxis(x, (mvaxis, traxis), (-2, -1))

    return x


def nd_shape_checking(x, y, mvaxis, traxis):
    """Check that the shape between two ndarray is consitent.

    x.shape = (nx_1, ..., n_xn, x_mvaxis, traxis)
    y.shape = (nx_1, ..., n_xn, y_mvaxis, traxis)
    """
    assert x.ndim == y.ndim
    assert all([k == i for num, (k, i) in enumerate(
                zip(x.shape, y.shape)) if num != mvaxis])


def nd_mi_gg(x, y, mvaxis=None, traxis=-1, biascorrect=True, demeaned=False,
             shape_checking=True):
    """Multi-dimentional MI between two Gaussian variables in bits.

    Parameters
    ----------
    x, y : array_like
        Arrays to consider for computing the Mutual Information. The two input
        variables x and y should have the same shape except on the mvaxis
        (if needed).
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI
    demeaned : bool | False
        Specifies whether the input data already has zero mean (true if it has
        been copula-normalized)
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False

    Returns
    -------
    mi : array_like
        The mutual information with the same shape as x and y, without the
        mvaxis and traxis
    """
    # Multi-dimentional shape checking
    if shape_checking:
        x = nd_reshape(x, mvaxis=mvaxis, traxis=traxis)
        y = nd_reshape(y, mvaxis=mvaxis, traxis=traxis)
        nd_shape_checking(x, y, mvaxis, traxis)

    # x.shape (..., x_mvaxis, traxis)
    # y.shape (..., y_mvaxis, traxis)
    ntrl = x.shape[-1]
    nvarx, nvary = x.shape[-2], y.shape[-2]
    nvarxy = nvarx + nvary

    # joint variable along the mvaxis
    xy = np.concatenate((x, y), axis=-2)
    if not demeaned:
        xy -= xy.mean(axis=-1, keepdims=True)
    cxy = np.einsum('...ij, ...kj->...ik', xy, xy)
    cxy /= float(ntrl - 1.)

    # submatrices of joint covariance
    cx = cxy[..., :nvarx, :nvarx]
    cy = cxy[..., nvarx:, nvarx:]

    # Cholesky decomposition
    chcxy = np.linalg.cholesky(cxy)
    chcx = np.linalg.cholesky(cx)
    chcy = np.linalg.cholesky(cy)

    # entropies in nats
    # normalizations cancel for mutual information
    hx = np.log(np.einsum('...ii->...i', chcx)).sum(-1)
    hy = np.log(np.einsum('...ii->...i', chcy)).sum(-1)
    hxy = np.log(np.einsum('...ii->...i', chcxy)).sum(-1)

    ln2 = np.log(2)
    if biascorrect:
        vec = np.arange(1, nvarxy + 1)
        psiterms = psi((ntrl - vec).astype(np.float) / 2.0) / 2.0
        dterm = (ln2 - np.log(ntrl - 1.0)) / 2.0
        hx -= nvarx * dterm - psiterms[:nvarx].sum()
        hy -= nvary * dterm - psiterms[:nvary].sum()
        hxy -= nvarxy * dterm - psiterms[:nvarxy].sum()

    # MI in bits
    i = (hx + hy - hxy) / ln2
    return i


def nd_gccmi_ccd(x, y, z, mvaxis=None, traxis=-1, gcrn=True,
                 shape_checking=True):
    """Multi-dimentional Gaussian-Copula conditional mutual information.

    This function performs a GC-CMI between 2 continuous variables conditioned
    on a discrete variable.

    Parameters
    ----------
    x, y : array_like
        Arrays to consider for computing the Mutual Information. The two input
        variables x and y should have the same shape except on the mvaxis
        (if needed).
    z : array_like
        Array that describes the conditions across the trial axis. Should be an
        array of shape (n_trials,) of integers (e.g. [0, 0, ..., 1, 1, 2, 2])
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    gcrn : bool | True
        Apply a Gaussian Copula rank normalization. This operation is
        relatively slow for big arrays.
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False

    Returns
    -------
    cmi : array_like
        Conditional mutual-information with the same shape as x and y without
        the mvaxis and traxis
    """
    # Multi-dimentional shape checking
    if shape_checking:
        x = nd_reshape(x, mvaxis=mvaxis, traxis=traxis)
        y = nd_reshape(y, mvaxis=mvaxis, traxis=traxis)
        nd_shape_checking(x, y, mvaxis, traxis)
    ntrl = x.shape[-1]

    # Find unique z elements
    _, idx = np.unique(z, return_index=True)
    zm = z[np.sort(idx)]
    sh = x.shape[:-3] if isinstance(mvaxis, int) else x.shape[:-2]
    zm_shape = list(sh) + [len(zm)]

    # calculate gcmi for each z value
    icond = np.zeros(zm_shape, dtype=float)
    pz = np.zeros((len(zm),), dtype=float)
    for num, zi in enumerate(zm):
        idx = z == zi
        pz[num] = idx.sum()
        if gcrn:
            thsx = copnorm(x[..., idx])
            thsy = copnorm(y[..., idx])
        else:
            thsx = x[..., idx]
            thsy = y[..., idx]
        icond[..., num] = nd_mi_gg(thsx, thsy, mvaxis=mvaxis, traxis=traxis,
                                   biascorrect=True, demeaned=True,
                                   shape_checking=False)
    pz /= ntrl

    # conditional mutual information
    cmi = np.sum(pz * icond, axis=-1)
    return cmi
