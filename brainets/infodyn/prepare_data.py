"""Prepare the data before computing the GCMI."""
import logging

import numpy as np
from xarray import DataArray, concat
import mne

from brainets.gcmi import copnorm

logger = logging.getLogger('brainets')


def gcmi_prepare_data(data, dp, roi, times=None, gcrn=True, aggregate='mean',
                      modality='meg', verbose=None):
    """Prepare the M/SEEG data before computing the GCMI.

    This function performs the following steps :

        * convert the list of inputs to standard NumPy arrays
        * mean channels that belong to the same roi per subject
        * apply a gaussian copula rank normalization
        * re-organize the output by roi

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
    aggregate : {'mean', 'concat'}
        Strategy to group sensors / channels inside an ROI. Choose either
        'mean' (e.g. mean HGA inside the roi) or 'concat' to concatenate all
        sites inside this roi

    Returns
    -------
    data : list
        List of DataArray organize by roi
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
    else:
        raise ValueError("aggregate shoud either be 'mean' or 'concat'")
    # gaussian copula rank normalization
    if gcrn:
        logger.info("    Apply the Gaussian Copula rank normalization")
        for d in data:
            d.data = copnorm(d.data)
    # reorganize the data by roi instead of subject
    logger.info("    Re-organize by roi")
    rois = [d.roi.values.tolist() for d in data]
    u_rois = np.unique(np.r_[tuple(rois)])
    x = []
    for r in u_rois:
        _x, _s, _s_id, q = [], [], [], 0
        for d in data:
            if r in d.roi.values:  # find if the ROI is present (sEEG)
                _x += [d.loc[:, r, :]]
                _s += [d.name]
                _s_id += [q] * d.shape[0]
                q += 1
        # Concatenate DataArray and attach additional attributes
        _c = concat(_x, dim='dp')
        _c.name = r
        _c.attrs['subjects'] = _s
        _c.attrs['z'] = np.array(_s_id)
        x += [_c]
    return x


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


if __name__ == '__main__':

    def generate_data(n_trials, n_channels, n_pts, n_roi=3):  # noqa
        x = np.random.rand(n_trials, n_channels, n_pts)
        dp = np.arange(n_trials)
        # dp = np.random.rand(n_trials)
        roi = ['roi%i' % k for k in np.random.randint(0, n_roi, n_channels)]
        times = np.linspace(-1.4, 1.4, n_pts, endpoint=True)
        return x, dp, roi, times

    # -------------------------------------------------------------------------
    # Dataset 1
    x_1, dp_1, roi_1, times = generate_data(50, 20, 100)
    # Dataset 2
    x_2, dp_2, roi_2, times = generate_data(47, 37, 100, 5)
    # Concatenate datasets
    x = [x_1, x_2]
    dp = [dp_1, dp_2]
    roi = [roi_1, roi_2]
    data = gcmi_prepare_data(x, dp, roi, times=times, aggregate='mean')
    print(data[0].T)
    # -------------------------------------------------------------------------
