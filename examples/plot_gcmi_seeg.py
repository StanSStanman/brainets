"""
Compute GCMI on a random SEEG datasets
======================================

This example illustrates :

    * How to prepare the SEEG data
    * Compute the GCMI

For SEEG data, this example assumes that the ROI are labelled using MarsAtlas.
"""
import numpy as np

from brainets.infodyn import (gcmi_random_dataset, gcmi_prepare_data,
                              gcmi_corrected)

import matplotlib.pyplot as plt


###############################################################################
# Generate random datasets
###############################################################################
# Generate two random datasets (e.g. data coming from two subjects), then
# concatenate the datasets (data, dP and roi)

# Subject 1
data_1, dp_1, roi_1 = gcmi_random_dataset(n_trials=300, random_state=1,
                                          n_channels=30, n_roi=15)
# Subject 2
data_2, dp_2, roi_2 = gcmi_random_dataset(n_trials=400, random_state=2,
                                          n_channels=25, n_roi=10)

# Concatenate the data of the two subjects
data = [data_1, data_2]
dp = [dp_1, dp_2]
roi = [roi_1, roi_2]
times = np.linspace(-1.5, 1.5, data_1.shape[-1], endpoint=True)

###############################################################################
# Prepare the data
###############################################################################
# Prepare the data before computing the GCMI.

# Prepare the data
x = gcmi_prepare_data(data, dp, roi=roi, times=times, smooth=None)
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
# Plot the data, GCMI and p-values
###############################################################################

plt.subplot(211)
plt.plot(smooth_times, gcmi.T)
plt.xlabel('Times (s)'), plt.ylabel('au')
plt.title('GCMI per roi')

plt.subplot(212)
plt.plot(smooth_times, pvalues.T)
plt.xlabel('Times (s)'), plt.ylabel('p-values')
plt.title('P-values per roi')

plt.tight_layout()

plt.show()
