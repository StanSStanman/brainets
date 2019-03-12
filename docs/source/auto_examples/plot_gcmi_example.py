"""
Compute GCMI on random datasets
===============================

This example illustrates :

    * How to organize the data (M/SEEG)
    * How to prepare the data
    * Compute the GCMI
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
data_1, dp_1, roi_1 = gcmi_random_dataset(n_trials=30, random_state=1)
# Subject 2
data_2, dp_2, roi_2 = gcmi_random_dataset(n_trials=40, random_state=2)

# Concatenate the data of the three subjects
data = [data_1, data_2]
dp = [dp_1, dp_2]
roi = [roi_1, roi_2]
times = np.linspace(-1.5, 1.5, data_1.shape[-1], endpoint=True)

###############################################################################
# Prepare the data
###############################################################################
# Prepare the data before computing the GCMI.

# Prepare the data
x = gcmi_prepare_data(data, dp, roi, times=times)

# Mean inside each roi
data_roi = np.c_[tuple([k.data.mean(0) for k in x])].T

###############################################################################
# Compute the GCMI
###############################################################################

smooth = 5
stat = 'cluster'  # {'cluster', 'maxstat'}
correction = 'fdr'  # {'fdr', 'bonferroni', 'maxstat'}
alpha = .05
n_perm = 30

gcmi, pvalues = gcmi_corrected(x, n_perm=n_perm, smooth=smooth, stat='cluster',
                               decim=None, correction=correction, alpha=alpha)

###############################################################################
# Plot the data, GCMI and p-values
###############################################################################

plt.subplot(311)
plt.plot(times, data_roi.T)
plt.xlabel('Times (s)'), plt.ylabel('uV/hz')
plt.title('Data per roi')

plt.subplot(312)
times = np.linspace(-1.5, 1.5, gcmi.shape[1], endpoint=True)
plt.plot(times, gcmi.T)
plt.xlabel('Times (s)'), plt.ylabel('au')
plt.title('GCMI per roi')

plt.subplot(313)
plt.plot(times, pvalues.T)
plt.xlabel('Times (s)'), plt.ylabel('p-values')
plt.title('P-values per roi')

plt.tight_layout()

plt.show()