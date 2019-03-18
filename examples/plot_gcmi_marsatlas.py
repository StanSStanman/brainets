"""
Plot GCMI using MarsAtlas
=========================

This example illustrates how to plot the GCMI using MarsAtlas.
"""
import numpy as np

from brainets.infodyn import (gcmi_random_dataset, gcmi_prepare_data,
                              gcmi_corrected)
from brainets.plot import plot_marsatlas

import matplotlib.pyplot as plt


###############################################################################
# Generate random datasets
###############################################################################
# Generate two random datasets (e.g. data coming from two subjects), then
# concatenate the datasets (data, dP and roi)

kw = dict(n_channels=96, n_roi=96)

# Subject 1
data_1, dp_1, roi_1 = gcmi_random_dataset(n_trials=30, random_state=1, **kw)
# Subject 2
data_2, dp_2, roi_2 = gcmi_random_dataset(n_trials=40, random_state=2, **kw)

# Concatenate the data of the two subjects
data = [data_1, data_2]
dp = [dp_1, dp_2]
times = np.linspace(-1.5, 1.5, data_1.shape[-1], endpoint=True)
data_roi = np.concatenate((data_1, data_2)).mean(0)

###############################################################################
# Prepare the data
###############################################################################
# Prepare the data before computing the GCMI.

# Prepare the data
x = gcmi_prepare_data(data, dp, times=times, smooth=None)
smooth_times = list(x[0].times.values)

###############################################################################
# Compute the GCMI
###############################################################################

stat = 'cluster'  # {'cluster', 'maxstat'}
correction = 'bonferroni'  # {'fdr', 'bonferroni', 'maxstat'}
alpha = .05
n_perm = 100

gcmi, pvalues = gcmi_corrected(x, n_perm=n_perm, stat=stat, alpha=alpha,
                               correction=correction, n_jobs=1)

###############################################################################
# Compute the GCMI
###############################################################################

# Remove non-significant GCMI
gcmi[pvalues > 0.01] = 0.
# Plot using MarsAtlas
plot_marsatlas(gcmi.T)
plt.show()
