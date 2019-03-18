"""Generate a random dataset."""
import logging

import numpy as np
from scipy.signal import savgol_filter

from brainets.io import load_marsatlas, set_log_level


logger = logging.getLogger('brainets')


def gcmi_random_dataset(n_trials=10, n_pts=100, n_channels=10, n_roi=3,
                        clusters=[(10, 30), (50, 70)], random_state=None,
                        verbose=None):
    """Generate a random dataset to test GCMI related functions.

    This function can be used to test or to check if the preparation and
    GCMI computations works.

    Parameters
    ----------
    n_trials : int | 30
        Number of trials
    n_pts : int | 200
        Number of time points
    n_channels : int | 10
        Number of channels
    n_roi : int | 3
        Number of ROI
    clusters : list | [(30, 70), (90, 130), (170, 190)]
        Index where to define clusters
    random_state : int | None
        Random state (use it for reproducibility)

    Returns
    -------
    data : array_like
        Random data to use of shape (n_trials, n_channels, n_pts)
    dp : array_like
        Behavioral array of shape (n_trials,)
    roi : array_like
        List of ROI names (using MarsAtlas)
    """
    logger.info("Generate a random dataset of shape (n_trials, n_channels, "
                "n_times) = (%i, %i, %i)" % (n_trials, n_channels, n_pts))
    set_log_level(verbose)
    if not isinstance(random_state, int):
        rnd = np.random
    else:
        rnd = np.random.RandomState(random_state)

    # -------------------------------------------------------------------------
    # Pick random n_roi
    logger.info("    Use %i ROIs" % n_roi)
    ma = load_marsatlas()
    pick_roi = rnd.randint(0, n_roi, (n_channels),)
    roi = np.array(ma['LR_Name'])[pick_roi]

    # -------------------------------------------------------------------------
    # Built a random dataset
    data = 100. * rnd.rand(n_trials, n_channels, n_pts)
    data = savgol_filter(data, 25, 3, axis=2)
    dp = np.sort(rnd.uniform(-1., 1., (n_trials)))

    # Introduce a correlation between the data and dp
    for c in clusters:
        _pow = rnd.randint(1, 3, 1)
        data[..., c[0]:c[1]] *= dp.reshape(-1, 1, 1) ** _pow

    return data, dp, roi
