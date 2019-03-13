"""Plot behavioral data."""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from brainets.behavior import load_behavioral


def plot_behavioral(report, files, variables=['edP', 'uedP', 'surprise',
                    'cho_uncho', 'kld']):
    """Plot behavioral data.

    This function can be used to summarize behavioral variables across subjects
    (it uses the output of the :func:`brainets.behavior.behavioral_analysis`)

    Note that this function uses the seaborn library to generate the plots

    Parameters
    ----------
    report : str
        Name of the pdf file (e.g. 'behavioral_report.pdf')
    files : dict
        Files to load. Should be a dictionary (e.g.
        {'suj_1': 'suj1_beh.xlsx', 'suj_2': 'suj2_beh'})
    variables : list | ['edP', 'uedP', 'surprise', 'cho_uncho', 'kld']
        List of variable names to plot.

    See also
    --------
    brainets.behavior.behavioral_analysis
    """
    assert isinstance(report, str) and '.pdf' in report
    assert isinstance(files, dict)
    import seaborn as sns
    # Loop over subjects / files
    with PdfPages(report) as pdf:
        for s, f in files.items():
            # Concatenate results across teams
            _, beh = load_behavioral(f, concat=True)
            # Generate figures
            fig, axes = plt.subplots(len(variables), 1, figsize=(8, 12))
            fig.suptitle(s)
            for num, v in enumerate(variables):
                plt.sca(axes[num])
                sns.relplot(x='index', y=v, hue='Team', data=beh, sizes=400,
                            alpha=.5, ax=axes[num], legend=False)
                plt.close(2)
                # Remove borders
                axes[num].spines['top'].set_visible(False)
                axes[num].spines['right'].set_visible(False)
                axes[num].spines['bottom'].set_visible(False)
                axes[num].spines['left'].set_visible(False)
                # Remove ticks
                if num + 1 != len(variables):
                    plt.xlabel('')
                    plt.tick_params(axis='x', which='both', bottom=False,
                                    top=False, labelbottom=False)
                else:
                    plt.xlabel('Trials')
            pdf.savefig(fig, dpi=600)
            plt.close()
