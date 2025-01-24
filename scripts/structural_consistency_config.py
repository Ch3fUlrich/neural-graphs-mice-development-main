"""Configuration for scripts/structural_consistency.py.

Use this file to configure parameters that cannot be passed as command line arguments to the script.
"""

# Number of random trials for computing structural consistency
NUM_TRIALS = 20

# Number of graph subsampling trials. Each of those will run the summary computation
# NUM_TRIALS times. This is bypassed if SUBSAMPLE_SIZE is set to None
NUM_SUBSAMPLE_TRIALS = 20

# Number of nodes to subsample from the graphs. Set it to None to use all nodes
SUBSAMPLE_SIZE = 250
