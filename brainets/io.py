"""I/O functions."""
import os
import logging

import numpy as np
import pandas as pd

import mne

from brainets.syslog import set_log_level

logger = logging.getLogger('brainets')


def get_data_path(file=None):
    """Get the path to brainets/data/.

    Alternatively, this function can also be used to load a file inside the
    data folder.

    Parameters
    ----------
    file : str
        File name

    Returns
    -------
    path : str
        Path to the data folder if file is None otherwise the path to the
        provided file.
    """
    pwd = os.path.dirname(os.path.realpath(__file__))
    file = file if isinstance(file, str) else ''
    return os.path.join(pwd, 'data', file)


def load_marsatlas():
    """Get the MarsAtlas dataframe.

    Returns
    -------
    df : DataFrame
        The MarsAtlas as a pandas DataFrame
    """
    ma_path = get_data_path('MarsAtlas_2015.xls')
    df = pd.read_excel(ma_path).iloc[:-1]
    df["LR_Name"] = df["Hemisphere"].map(str) + ['_'] * len(df) + df["Name"]
    return df


###############################################################################
#                             MNE CONVERSION
###############################################################################

def mne_epochstfr_to_epochs(epoch, freqs=None, verbose=None):
    """Convert an MNE EpochsTFR to Epochs instance.

    Parameters
    ----------
    epoch : mne.time_frequency.EpochsTFR | str
        Should either be an EpochsTFR instance or a path to a -tfr.h5 file
    freqs : tuple, list | None
        The frequencies to select Use None to select all frequencies or a tuple
        of two floats to select a sub-band. The final Epochs instance is
        obtained by taking the mean across selected frequencies.

    Returns
    -------
    r_epoch : mne.Epochs
        Epochs instance
    """
    set_log_level(verbose)
    if isinstance(epoch, str):
        assert '-tfr.h5' in epoch, "File should end with -tfr.h5 file"
        epoch = mne.time_frequency.read_tfrs(epoch)[0]
    assert isinstance(epoch, mne.time_frequency.EpochsTFR)
    # Handle frequencies
    epoch_freqs = epoch.freqs
    if freqs is None:
        logger.info('    Selecting all frequencies')
        sl = slice(None)
    elif isinstance(freqs, (list, tuple, np.ndarray)):
        assert len(freqs) == 2, "`freqs` should be tuple of two elements"
        logger.info("    Selecting frequencies "
                    "(%.2f, %.2f)" % (freqs[0], freqs[1]))
        _idx = np.abs(epoch_freqs.reshape(-1, 1) - np.array(
            freqs).reshape(1, -1)).argmin(0)
        sl = slice(_idx[0], _idx[1])
    # Built the Epochs instance
    info = epoch.info
    data = epoch.data[..., sl, :].mean(2)
    return mne.EpochsArray(data, info, tmin=epoch.times[0], verbose=verbose)
