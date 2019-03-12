"""
Behavioral analysis for the CausaL task
=======================================

This script illustrate how to perform the behavioral analysis for the CausaL
task (M/SEEG)
"""
import numpy as np

from brainets.behavior import behavioral_analysis, load_dp

###############################################################################
# Organize your triggers
###############################################################################
# If you want to use the function :func:`brainets.behavior.behavioral_analysis`
# you've to organize the triggers in a specific way. First, let say that the
# subject performed n_trials = 600 (15 teams x 40 trials per team)

# Team trigger : the team trigger describe, for each trial, the team number
# that is played
n_trials_per_team = 40
team_order = [4, 11, 7, 15, 13, 3, 1, 12, 14, 5, 10, 2, 9, 6, 8]
tr_team = np.repeat(team_order, n_trials_per_team)

# Play / not play trigger : this trigger describes if the subject play (True)
# or not (False)
tr_play = np.random.randint(0, 2, (len(tr_team),)).astype(bool)

# Win / lose trigger : this trigger describes if the subject win (True) or
# lose (Flse)
tr_win = np.random.randint(0, 2, (len(tr_team),)).astype(bool)

print(np.c_[tr_team, tr_play, tr_win])

###############################################################################
# Run the behavioral analysis
###############################################################################
# Now you've the triggers for a single subject properly organized, you can run
# the behavioral analysis.

modality = 'meg'  # {'meg', 'seeg'}
save_as = 'subj1_behavioral_analysis.xlsx'
behavioral_analysis(tr_team, tr_play, tr_win, modality=modality,
                    save_as=save_as)

###############################################################################
# Load a variable in the generated table
###############################################################################