"""Prepare the data before computing the GCMI."""
import logging

import numpy as np
from xarray import DataArray, concat
import mne

from brainets.io import load_marsatlas, set_log_level
from brainets.gcmi import copnorm

logger = logging.getLogger('brainets')


def gcmi_prepare_data(data, dp, roi=None, times=None, smooth=None, decim=None,
                      aggregate='mean', modality='meg', verbose=None):
    """Prepare the M/SEEG data before computing the GCMI.

    This function performs the following steps :

        * convert the list of inputs to standard NumPy arrays
        * mean channels that belong to the same roi per subject
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
    roi : list | None
        List of arrays of shape (n_channels,) describing the ROI name of each
        channel. If None and if the modality is MEG, the ROI are inferred using
        MarsAtlas
    times : array_like | None
        The time vector. All of the subject should have the number of time
        points. If None, a default (-1.5, 1.5) secondes vector is created. If
        MNE instances are provided, the time vector is inferred from it
    smooth : int | None
        Time smoothing factor
    decim : int | None
        Decimation factor (use it to reduce the number of time points)
    aggregate : {'mean', 'concat'}
        Strategy to group sensors / channels inside an ROI. Choose either
        'mean' (e.g. mean HGA inside the roi) or 'concat' to concatenate all
        sites inside this roi

    Returns
    -------
    data : list
        List of DataArray organize by roi
    """
    set_log_level(verbose)
    # -------------------------------------------------------------------------
    # Input checking
    assert all([isinstance(k, (list, tuple)) for k in (data, dp)])
    assert len(data) == len(dp)
    n_suj = len(data)
    decim = 1 if not isinstance(decim, int) else decim
    assert decim > 0, "`decim` should be an integer > 0"
    logger.info("Prepare the %s data of %i subjects" % (modality, n_suj))

    # -------------------------------------------------------------------------
    # infer MEG roi
    if (roi is None) and (modality == 'meg'):
        logger.info("    Infer the ROI for the meg data")
        _roi = list(load_marsatlas()['LR_Name'])
        roi = [np.array(_roi) for k in range(len(data))]
    roi = [np.array(k) for k in roi]

    # -------------------------------------------------------------------------
    # extract numpy arrays from files
    if isinstance(smooth, int):
        logger.info("    Temporal smoothing of %i points with a %i points "
                    "decimation" % (smooth, decim))
    # check data type and convert it to numpy array
    data = [_prepare_single_subject(k, i, j, times, n, smooth, decim) for n, (
            k, i, j) in enumerate(zip(data, dp, roi))]

    # -------------------------------------------------------------------------
    # Define the strategy to concatenate the data
    same_shape = all([data[0].shape[:-1] == k.shape[:-1] for k in data])
    same_roi = all([np.array_equal(k, roi[0]) for k in roi])
    if same_shape and same_roi:
        logger.info("    Concatenate the data along the trial axis")
        z = [np.array([num] * k.shape[-1]) for num, k in enumerate(data)]
        # Concatenate the data across subjects
        c = concat(data, dim='dp')
        c.attrs['z'] = np.r_[tuple(z)]
        c.attrs['modality'] = modality
        data = []
        for r in roi[0]:
            _data = c.loc[r, ...]
            _data.name = r
            data += [_data]
            del _data
        return data

    # -------------------------------------------------------------------------
    # agregate the data inside ROI
    if aggregate == 'mean':
        logger.info("    Take the mean inside each ROI")
        data = [k.groupby('roi').mean('roi') for k in data]
    elif aggregate == 'concat':
        raise NotImplementedError()
    else:
        raise ValueError("aggregate shoud either be 'mean' or 'concat'")

    # -------------------------------------------------------------------------
    # reorganize the data by roi instead of subject
    logger.info("    Re-organize by roi")
    rois = [d.roi.values.tolist() for d in data]
    u_rois = np.unique(np.r_[tuple(rois)])
    x = []
    for r in u_rois:
        _x, _s, _s_id, q = [], [], [], 0
        for d in data:
            if r in d.roi.values:  # find if the ROI is present (sEEG)
                _x += [d.loc[r, ...]]
                _s += [d.name]
                _s_id += [q] * d.shape[-1]
                q += 1
        # Concatenate DataArray and attach additional attributes
        _c = concat(_x, dim='dp')
        _c.name = r
        _c.attrs['subjects'] = _s
        _c.attrs['z'] = np.array(_s_id)
        _c.attrs['modality'] = modality
        x += [_c]
    return x


def _prepare_single_subject(x, dp, roi, times, n, smooth, decim):
    """Prepare the data of a single subject."""
    # -------------------------------------------------------------------------
    # Load the MNE-instance
    if isinstance(x, str) and ('-tfr.h5' in x):     # EpochsTFR instance
        x = mne.time_frequency.read_tfrs(x)[0]
        times = x.times
    elif isinstance(x, str) and ('-epo.fif' in x):  # Epochs instance
        x = mne.read_epochs(x)
        times = x.times

    # -------------------------------------------------------------------------
    # Check inputs
    if isinstance(x, (mne.Epochs, mne.EpochsArray, mne.epochs.EpochsFIF,
                      mne.time_frequency.EpochsTFR, mne.epochs.BaseEpochs)):
        data = x.get_data()
    elif isinstance(x, np.ndarray):
        data = x
    else:
        raise TypeError("x of type %s not supported" % str(type(x)))

    # -------------------------------------------------------------------------
    # Handle multi-dimentional arrays
    if data.ndim == 4:  # TF : (n_trials, n_channels, n_freqs, n_pts)
        if data.shape[2] == 1:
            data = data[..., 0, :]
        else:
            data = data.mean(2)
            logger.warning("Multiple frequencies detected. Take the mean "
                           "across frequencies")
    assert (data.ndim == 3)

    # -------------------------------------------------------------------------
    # time vector construction
    if times is None:
        times = np.linspace(-1.5, 1.5, data.shape[-1], endpoint=True)

    # temporal smoothing
    if isinstance(smooth, int):
        ud, ut = data.copy(), times.copy()
        del data, times
        vec = np.arange(smooth, len(ut) - smooth, decim)
        times = np.zeros_like(vec, dtype=float)
        dp = np.tile(dp.reshape(-1, 1), (2 * smooth + 1)).ravel(order='F')
        data = np.zeros((len(dp), ud.shape[1], len(vec)), dtype=float)
        for r in range(data.shape[1]):
            for nv, v in enumerate(vec):
                sl = slice(v - smooth, v + smooth + 1)
                data[:, r, nv] = ud[:, r, sl].ravel(order='F')
                times[nv] = ut[sl].mean()

    # data.reshape(n_roi, n_times, mvaxis, n_trials)
    data = np.moveaxis(data, 0, -1)[..., np.newaxis, :]
    assert data.shape == (len(roi), len(times), 1, len(dp))

    # -------------------------------------------------------------------------
    # Gaussian copula rank normalization
    dp, data = copnorm(dp), copnorm(data)

    # -------------------------------------------------------------------------
    # DataArray conversion
    data = DataArray(data.astype(np.float32), coords=[roi, times, [0], dp],
                     name='subject%i' % n,
                     dims=['roi', 'times', 'mvaxis', 'dp'])
    return data


if __name__ == '__main__':

    def generate_data(n_trials, n_channels, n_pts, n_roi=3):  # noqa
        x = np.random.rand(n_trials, n_channels, n_pts)
        # dp = np.arange(n_trials)
        dp = np.random.rand(n_trials)
        roi = ['roi%i' % k for k in np.random.randint(0, n_roi, n_channels)]
        times = np.linspace(-1.4, 1.4, n_pts, endpoint=True)
        return x, dp, roi, times

    # -------------------------------------------------------------------------
    # Dataset 1
    x_1, dp_1, roi_1, times = generate_data(30, 96, 100)
    # Dataset 2
    x_2, dp_2, roi_2, times = generate_data(50, 96, 100)
    # Concatenate datasets
    x = [x_1, x_2]
    dp = [dp_1, dp_2]
    roi = [roi_1, roi_1]
    data = gcmi_prepare_data(x, dp, times=times, aggregate='mean', smooth=None)
    print(data)
    # -------------------------------------------------------------------------
