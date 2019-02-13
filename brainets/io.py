"""I/O functions."""
import os

import pandas as pd


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
