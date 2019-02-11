"""Behavioral functions."""
import os
import logging

import numpy as np
import pandas as pd
from scipy import special

from brainets.syslog import set_log_level

logger = logging.getLogger('brainets')


def pdf(a, b, n_pts=100):
    """Generate a beta distribution from float two inputs.

    Parameters
    ----------
    a : int | float
        Alpha parameter
    b : int | float
        Beta parameter
    n_pts : int | 100
        Number of points composing the beta distribution. Alternatively, you
        can also gives your own vector.

    Returns
    -------
    z : array_like
        The beta distribution of shape (n_pts,)
    """
    assert isinstance(a, (int, float)) and isinstance(b, (int, float))
    # Function to generate a beta distribution with a fixed number of steps
    if isinstance(n_pts, int):
        n_pts = np.linspace(0., 1., n_pts, endpoint=True)
    z = special.beta(a, b)
    return (n_pts ** (a - 1) * ((1 - n_pts) ** (b - 1)) / z)


def bincumsum(x):
    """Cumulative sum for entries of 0 and 1.

    This function uses np.cumsum but force the first value to be 1.

    Parameters
    ----------
    x : array_like
        Vector array filled with 0 and 1 of shape (n_trials,)

    Returns
    -------
    cumsum : array_like
        Cumulative sum of shape (n_trials)
    """
    x = np.asarray(x, dtype=int).copy()
    is_valid = (0 <= x.min() <= 1) and (0 <= x.max() <= 1)
    assert is_valid, "x should only contains 0 and 1"
    # x[0] = 1  # ensure first element is 1
    return np.cumsum(x) + 1.1


def get_causal_probabilities(as_array=False, modality='meg'):
    """Get the probabilities per team used in the CausaL task.

    Parameters
    ----------
    as_array : bool | True
        Get the probabilities as an array of shape (15, 4) where 4 refers to
        team number, win, lose and dP. Alternatively, if False, get a pandas
        dataframe
    modality : {'meg', 'seeg'}
        Because the probabilities are different between the meg and seeg task,
        you have to specify the recording modality.

    Returns
    -------
    df : array_like | pd.DataFrame
        The probabilities of the CausaL task.
    """
    assert modality in ['meg', 'seeg'], ("`modality` should either be 'meg' or"
                                         " 'seeg'")
    team = np.linspace(1, 15, 15, endpoint=True, dtype=int)
    logger.info('    - Get %s probabilities' % modality)
    if modality == 'meg':
        win = [.7, .4, .7, .2, .4, .1, .6, .8, .6, .3, .9, .3, .5, .8, .5]
        lose = [.4, .7, .1, .8, .4, .7, .3, .5, .6, .9, .3, .6, .8, .2, .5]
    elif modality == 'seeg':
        win = [.5, .5, .8, .8, .8, .6, .6, .7, .7, 1, .4, .7, .6, .9, .9]
        lose = [.5, .3, .4, .2, 0, .6, .4, .3, .1, .2, .4, .5, .2, .3, .1]
    dp = np.array(win) - np.array(lose)
    if as_array:
        return np.c_[team, win, lose, dp]
    else:
        return pd.DataFrame({'Team': team, 'P(O|A)': win, 'P(O|nA)': lose,
                             'dP': dp})
        # return pd.DataFrame(dict(Team=team, Win=win, Lose=lose, dP=dp))


def behavioral_analysis(tr_team, tr_play, tr_win, save_as=None,
                        embedded_plot=True, modality='meg', verbose=None):
    """Perform behavioral analysis using team, play and win triggers.

    Parameters
    ----------
    tr_team : array_like
        Array describing the team number per trial (e.g. [6, 6, 6, ..., 15])
    tr_play : array_like
        Array describing if the subject is playing (1) or not (0)
    tr_win : array_like
        Array describing if the subject win (1) or lose (0)
    save_as : string | None
        Full path to a .xlsx file where to save the Excel file
    embedded_plot : bool | True
        Put P(O|A), P(O|nA) and dP plots inside the excel file.
    modality : {'meg', 'seeg'}
        Because the probabilities are different between the meg and seeg task,
        you have to specify the recording modality.

    Returns
    -------
    summary : dataframe
        A pandas dataframe that summarize the estimated probabilities (edP)
        with theorical values define by the task (different if seeg or meg)
    behavior : dict
        Dictionary organize by team number in which conditional probabilities
        and cumulative sum are saved.
    """
    set_log_level(verbose)
    # Sanity check
    (tr_team, tr_play, tr_win) = tuple([np.asarray(k, dtype=int) for k in [
                                       tr_team, tr_play, tr_win]])
    assert tr_team.shape == tr_play.shape == tr_win.shape
    _u = all([np.array_equal(np.unique(k), [0, 1]) for k in [tr_play, tr_win]])
    assert _u, "`tr_play` and `tr_win` should only contains 0 and 1"
    is_t_min, is_t_max = 1 <= tr_team.min() <= 15, 1 <= tr_team.max() <= 15
    assert is_t_min and is_t_max, "Team number must be between [1, 15]"
    # Boolean analysis
    logger.info('    - Get conditional booleans')
    is_oa = np.logical_and(tr_win == 1, tr_play == 1).astype(int)
    is_noa = np.logical_and(tr_win == 0, tr_play == 1).astype(int)
    is_ona = np.logical_and(tr_win == 1, tr_play == 0).astype(int)
    is_nona = np.logical_and(tr_win == 0, tr_play == 0).astype(int)
    # Compute the cumulative sum per team
    behavior = dict()
    col = ['Team', 'Play', 'Win', 'O|A', 'nO|A', 'O|nA', 'nO|nA', 'f(O|A)',
           'f(nO|A)', 'f(O|nA)', 'f(nO|nA)', 'eP(O|A)', 'eP(O|nA)', 'edP']
    logger.info('    - Split cumulative sum per team')
    for team in np.unique(tr_team):
        _df = dict()
        idx_team = tr_team == team
        # Retains trigger for the current team
        _df['Team'] = tr_team[idx_team]
        _df['Play'], _df['Win'] = tr_play[idx_team], tr_win[idx_team]
        # Retains the probability
        _df['O|A'], _df['O|nA'] = is_oa[idx_team], is_ona[idx_team]
        _df['nO|A'], _df['nO|nA'] = is_noa[idx_team], is_nona[idx_team]
        # Get the cumulative sum for each condition
        _df['f(O|A)'] = bincumsum(_df['O|A'])
        _df['f(O|nA)'] = bincumsum(_df['O|nA'])
        _df['f(nO|A)'] = bincumsum(_df['nO|A'])
        _df['f(nO|nA)'] = bincumsum(_df['nO|nA'])
        # Compute P(O|A) and P(O|nA)
        _df['eP(O|A)'] = _df['f(O|A)'] / (_df['f(O|A)'] + _df['f(nO|A)'])
        _df['eP(O|nA)'] = _df['f(O|nA)'] / (_df['f(O|nA)'] + _df['f(nO|nA)'])
        _df['edP'] = _df['eP(O|A)'] - _df['eP(O|nA)']
        behavior[team] = pd.DataFrame(_df, columns=col)
    # Summary table
    logger.info("    - Summary table")
    summary = get_causal_probabilities(False, modality)
    _task = np.full((len(summary), 3), np.nan)
    for t in np.unique(tr_team) - 1:
        _task[t, 0] = behavior[t + 1]['eP(O|A)'].iloc[-1]
        _task[t, 1] = behavior[t + 1]['eP(O|nA)'].iloc[-1]
        _task[t, 2] = behavior[t + 1]['edP'].iloc[-1]
    summary['eP(O|A)'] = _task[:, 0]
    summary['eP(O|nA)'] = _task[:, 1]
    summary['edP'] = _task[:, 2]
    # Change the column order for plotting
    col = ['Team', 'P(O|A)', 'eP(O|A)', 'P(O|nA)', 'eP(O|nA)', 'dP', 'edP']
    summary = summary[col]
    # Save the dataframe
    if isinstance(save_as, str):
        with pd.ExcelWriter(save_as) as writer:
            summary.to_excel(writer, sheet_name='Summary')
            for team, df in behavior.items():
                df.to_excel(writer, sheet_name='Team %i' % team)
        # Generate plots inside the Excel file
        from openpyxl import load_workbook
        from openpyxl.chart import Reference, LineChart, BarChart
        wb = load_workbook(save_as)
        # Summary plot
        ws = wb['Summary']
        c1 = BarChart()
        title = "Comparison between dP and estimated dP (edP)"
        _xl_plot(c1, title, 'Team', 'Contingency')
        data = Reference(ws, min_col=7, min_row=1, max_row=17, max_col=8)
        team = Reference(ws, min_col=2, min_row=1, max_row=17)
        c1.set_categories(team)
        c1.add_data(data, titles_from_data=True)
        ws.add_chart(c1, "A18")
        # Team plot
        for team, df in behavior.items():
            ws = wb['Team %i' % team]
            c1 = LineChart()
            title = "Contingency evolution across trials"
            _xl_plot(c1, title, 'Trials', 'dP')
            data = Reference(ws, min_col=13, max_col=15, min_row=1, max_row=41)
            c1.add_data(data, titles_from_data=True)
            ws.add_chart(c1, "A45")
        wb.save(save_as)
        logger.info("    - Behavioral analysis saved to %s" % save_as)

    return summary, behavior


def load_behavioral(path, verbose=None):
    """Load the behavioral analysis excel file of a single subject.

    The purpose of this function is to load the excel file that have been
    generated using the function `behavioral_analysis`.

    Parameters
    ----------
    path : str
        Full path to the excel file

    Returns
    -------
    summary : pandas.DataFrame
        The dataframe that summarize probabilities and contingency.
    behavior : dict
        A dictionary where the keys refer to the team number. Items are
        dataframes with all of the info per trial.
    """
    assert os.path.isfile(path)
    set_log_level(verbose)
    logger.info('Loading %s' % path)
    xl = pd.ExcelFile(path)
    sheet_names = xl.sheet_names
    assert sheet_names[0] == 'Summary'
    # Get the summary
    logger.info('    - Reading summary')
    summary = xl.parse('Summary')
    # Read team sheet
    behavior = dict()
    logger.info('    - Reading team')
    for s in sheet_names[1::]:
        behavior[int(s.split('Team ')[1])] = xl.parse(s)
    return summary, behavior


def _xl_plot(c1, title='', xlabel='', ylabel='', height=15, width=25):
    """Quick and dirty excel plot fix."""
    c1.title = title
    c1.y_axis.title = ylabel
    c1.x_axis.title = xlabel
    c1.y_axis.delete = False
    c1.height = height
    c1.width = width
