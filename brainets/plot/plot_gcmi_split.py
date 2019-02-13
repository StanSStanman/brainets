"""2d plot of the gcmi."""
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

from brainets.io import load_marsatlas
from brainets.syslog import set_log_level

logger = logging.getLogger('brainets')


def plot_gcmi_split(data, time=None, modality='meg', seeg_roi=None, contrast=5,
                    cmap='viridis', title=None, verbose=None):
    """Plot GCMI in splitted subplots, sorted using MarsAtlas.

    Parameters
    ----------
    data : array_like
        GCMI result across ROI of shape (n_pts, n_roi)
    time : list | tuple | None
        Time boundaries. Should be (time_start, time_end). If None, a default
        time vector is set between (-1.5, 1.5)
    modality : {'meg', 'seeg'}
        The recording modality. Should either be 'meg' or 'seeg'.
    seeg_roi : pd.DataFrame | None
        The ROI dataframe in case of sEEG data. Should contains n_rois rows
        and a MarsAtlas column
    contrast : int | float
        Contrast to use for the plot. A contrast of 5 means that vmin is set to
        5% of the data and vmax 95% of the data
    title : string | None
        Title of the figure
    cmap : string | 'viridis'
        The colormap to use

    Returns
    -------
    fig_l, fig_r : plt.figure
        Figures for the left and right hemisphere
    """
    set_log_level(verbose)
    assert modality in ['meg', 'seeg']
    assert isinstance(data, np.ndarray) and (data.ndim == 2)

    # Load MarsAtlas DataFrame
    logger.info('    Load MarsAtlas labels')
    df_ma = load_marsatlas()

    # Prepare the data before plotting according to the recording modality
    logger.info('    Prepare the data for %s modality' % modality)
    if modality == 'meg':
        assert data.shape[1] == len(df_ma), ("`data` should have a shape of "
                                             "(n_pts, %i)" % len(df_ma))
        df, df_ma = _prepare_data_meg(df_ma, data)
    elif modality == 'seeg':
        assert isinstance(seeg_roi, pd.DataFrame) and (data.shape[1] == len(
            seeg_roi)), ("`data` should have a shape of "
                         "(n_pts, %i)" % len(seeg_roi))
        df, df_ma = _prepare_data_seeg(df_ma, data, seeg_roi)

    # Built the multi-indexing
    assert len(df.columns) == len(df_ma)
    mi = pd.MultiIndex.from_frame(df_ma[['Hemisphere', 'Lobe', 'Name']])
    df.columns = mi

    # Time vector
    if isinstance(time, (list, tuple, np.ndarray)) and (len(time) == 2):
        time = np.linspace(time[0], time[1], data.shape[0], endpoint=True)
        logger.info('    Generate time vector')
    else:
        time = np.linspace(-1.5, 1.5, data.shape[0], endpoint=True)
        logger.warning("Automatically generate a time vector between "
                       "(-1.5, 1.5)")

    # Get colorbar limits
    if isinstance(contrast, (int, float)):
        vmin = np.percentile(data, contrast)
        vmax = np.percentile(data, 100 - contrast)
    else:
        vmin, vmax = data.min(), data.max()
    kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)

    # Generate plots
    title = '' if not isinstance(title, str) else title
    fig_l = _plot_gcmi_hemi(df, 'L', time, title, **kwargs)
    fig_r = _plot_gcmi_hemi(df, 'R', time, title, **kwargs)
    return fig_l, fig_r


def _prepare_data_seeg(df_ma, data, seeg_roi):
    """Prepare sEEG data for plotting."""
    # Group roi by MarsAtlas
    gp = seeg_roi.groupby('MarsAtlas').groups
    # Mean data by ROI according to the order defined in the default MarsAtlas
    # dataframe
    missings, data_roi = [], []
    for num, n in enumerate(df_ma['LR_Name']):
        if n in gp.keys():
            # Mean data
            data_roi += [data[:, gp[n]].mean(1)]
        else:
            missings += [num]
            data_roi += [np.full((data.shape[0]), -1.)]
    # Display missing ROI
    miss = df_ma['LR_Name'].iloc[missings]
    logger.info('   Missing ROIs : %s' % ', '.join(miss))
    return pd.DataFrame(np.stack(data_roi).T), df_ma


def _prepare_data_meg(df_ma, data):
    """Prepare MEG data for plotting."""
    return pd.DataFrame(data), df_ma


def _plot_gcmi_hemi(df, hemi, time, title, **kwargs):
    """Plot for a single hemisphere."""
    fig = plt.figure()
    fig.suptitle('%s Hemisphere %s' % (title, hemi), fontweight='bold',
                 fontsize=15, y=1.)
    gs = GridSpec(10, 12)
    # Frontal areas
    ax1 = plt.subplot(gs[:8, :5])
    _plot_single_subplot(ax1, df, time, hemi, 'Frontal', **kwargs)
    # Occipital areas
    ax2 = plt.subplot(gs[8:, :5])
    _plot_single_subplot(ax2, df, time, hemi, 'Occipital', xticks=True,
                         **kwargs)
    # Temporal areas
    ax3 = plt.subplot(gs[:3, 5:-1])
    _plot_single_subplot(ax3, df, time, hemi, 'Temporal', **kwargs)
    # Parietal areas
    ax4 = plt.subplot(gs[3:7, 5:-1])
    _plot_single_subplot(ax4, df, time, hemi, 'Parietal', **kwargs)
    # Subcortical regions
    ax5 = plt.subplot(gs[7:, 5:-1])
    _plot_single_subplot(ax5, df, time, hemi, 'Subcortical', xticks=True,
                         **kwargs)
    # Colorbar
    ax6 = plt.subplot(gs[:, -1])
    norm = mpl.colors.Normalize(vmin=kwargs['vmin'], vmax=kwargs['vmax'])
    mpl.colorbar.ColorbarBase(ax6, cmap=kwargs['cmap'], norm=norm,
                              label='GCMI')
    # This is a bug of mpl, but plt.tight_layout doesn't work...
    fig.set_tight_layout(True)


def _plot_single_subplot(ax, df, time, hemi, lobe, xticks=False, **kwargs):
    """Generate the plot of a single subplot."""
    levels = df.keys().levels
    if (hemi not in levels[0]) or (lobe not in levels[1]):
        logger.warning("Nothing in the %s %s lobe" % (hemi, lobe))
        ax.axis('off')
        return None
    # Get the data for this hemisphere / lobe
    _df = df[hemi][lobe]
    data, yticks = np.array(_df), list(_df.keys())
    # Check if the data is not always the same
    if data.min() == data.max():
        ax.axis('off')
        return None
    # Make the plot
    n_t = len(yticks)
    yvec = np.arange(-1, n_t)  # I've no idea why mpl start at -1...
    ax.pcolormesh(time, yvec, data.T, **kwargs)
    plt.title('%s lobe' % lobe)
    plt.yticks(np.linspace(-.5, n_t - 1.5, n_t, endpoint=True))
    ax.set_yticklabels(yticks)
    plt.axvline(0, color='w', linewidth=3)
    # Remove xticks
    ax.tick_params(axis='both', which='both', length=0)
    if not xticks:
        plt.tick_params(axis='x', which='both', bottom=False, top=False,
                        labelbottom=False)
    # Remove borders
    for k in ['top', 'bottom', 'left', 'right']:
        ax.spines[k].set_visible(False)
