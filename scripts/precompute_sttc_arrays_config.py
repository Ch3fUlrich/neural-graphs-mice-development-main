"""
Configuration file for the script ./precompute_sttc_arrays.py
"""

# Random seed for reproducibility
seed = 2024

# Synchronicity window (in seconds) for the STTCs
dt = 0.25

# Method to generate surrogate STTC values
method = "pulse_shuffle"

# Number of surrogate STTC matrices to generate
num_trials = 3

# Number of jobs to run in parallel when generating surrogate STTC values
n_jobs = 1

# Whether to force recomputation of STTC arrays
force_recompute = False
