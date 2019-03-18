"""Fast implementation of GCMI."""
import numpy as np
import scipy as sp


def gccmi_ccd(x, y, z, zm):
    """GC CMI between 2 continuous variables conditioned on a discrete variable.

    I = gccmi_ccd(x,y,z,Zm) returns the CMI between two (possibly
    multidimensional) continuous variables, x and y, conditioned on a third
    discrete variable z, estimated via a Gaussian copula.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples first axis)
    z should contain integer values in the range [0 Zm-1] (inclusive).
    """
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    ntrl = x.shape[1]

    # calculate gcmi for each z value
    icond = np.zeros(zm, dtype=float)
    pz = np.zeros(zm, dtype=float)
    for zi in range(zm):
        idx = z == zi
        thsx = copnorm(x[:, idx])
        thsy = copnorm(y[:, idx])
        pz[zi] = x.shape[1]
        icond[zi] = mi_gg(thsx, thsy, True, True)

    pz /= float(ntrl)

    # conditional mutual information
    cmi = np.sum(pz * icond)
    return cmi


def ctransform(x):
    """Copula transformation (empirical CDF).

    cx = ctransform(x) returns the empirical CDF value along the first
    axis of x. Data is ranked and scaled within [0 1] (open interval).
    """
    xi = np.argsort(x)
    xr = np.argsort(xi).astype(float)
    xr += 1.
    xr /= float(xr.shape[-1] + 1)
    return xr


def copnorm(x):
    """Copula normalization.

    cx = copnorm(x) returns standard normal samples with the same empirical
    CDF value as the input. Operates along the last axis.
    """
    transform = ctransform(x)
    # cx = sp.stats.norm.ppf(transform)
    cx = sp.special.ndtri(transform)
    return cx


def mi_gg(x, y, biascorrect=True, demeaned=False):
    """Mutual information (MI) between two Gaussian variables in bits.

    I = mi_gg(x,y) returns the MI between two (possibly multidimensional)
    Gassian variables, x and y, with bias correction.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples last axis)

    biascorrect : true / false option (default true) which specifies whether
    bias correction should be applied to the esimtated MI.
    demeaned : false / true option (default false) which specifies whether th
    input data already has zero mean (true if it has been copula-normalized)
    """
    ntrl = x.shape[1]
    nvarx = x.shape[0]
    nvary = y.shape[0]
    nvarxy = nvarx + nvary

    # joint variable
    xy = np.vstack((x, y))
    if not demeaned:
        xy -= xy.mean(axis=1, keepdims=True)
    cxy = np.dot(xy, xy.T) / float(ntrl - 1)
    # submatrices of joint covariance
    cx = cxy[:nvarx, :nvarx]
    cy = cxy[nvarx:, nvarx:]

    chcxy = np.linalg.cholesky(cxy)
    chcx = np.linalg.cholesky(cx)
    chcy = np.linalg.cholesky(cy)

    # entropies in nats
    # normalizations cancel for mutual information
    hx = np.sum(np.log(np.diagonal(chcx)))
    hy = np.sum(np.log(np.diagonal(chcy)))
    hxy = np.sum(np.log(np.diagonal(chcxy)))

    ln2 = np.log(2)
    if biascorrect:
        vec = np.arange(1, nvarxy + 1)
        psiterms = sp.special.psi((ntrl - vec).astype(np.float) / 2.0) / 2.0
        dterm = (ln2 - np.log(ntrl - 1.0)) / 2.0
        hx -= nvarx * dterm - psiterms[:nvarx].sum()
        hy -= nvary * dterm - psiterms[:nvary].sum()
        hxy -= nvarxy * dterm - psiterms[:nvarxy].sum()

    # MI in bits
    i = (hx + hy - hxy) / ln2
    return i
