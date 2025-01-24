"""
Precompute STTC arrays for all sessions in the dataset.

Important: Make sure to set up the configuration file `precompute_sttc_arrays_config.py`
before running this script.
"""

import logging
import os
import traceback

import click
import numpy as np
import precompute_sttc_arrays_config as config
from joblib import Parallel, delayed

from ngmd import dataset, sttc, timeseries, utils

# Set up logging
logging.basicConfig(level=logging.INFO)


def compute_sttc_arrays(animal_dataset, session):
    force_recompute = config.force_recompute
    dt = config.dt
    method = config.method
    num_trials = config.num_trials
    n_jobs = config.n_jobs

    # Set save file path
    save_file_path = os.path.join(session.path, "sttc_arrays.npz")

    # Check if STTC arrays have already been computed for this session
    if os.path.exists(save_file_path):
        if not force_recompute:
            # Skip session if STTC arrays have already been computed
            return

    # Load binary traces
    binary_traces = animal_dataset.load_traces_from_good_rois(session)

    # Compute STTC matrix
    sttc_mat = sttc.matrix_spike_time_tiling_coefficient(
        binary_traces, session.sample_rate, dt
    )

    # Compute surrogate STTC values
    surrogate_sttc_values = timeseries.compile_surrogate_sttc_values(
        binary_traces,
        session.sample_rate,
        method=method,
        num_trials=num_trials,
        n_jobs=n_jobs,
        dt=dt,
    )

    # Compute percentiles matrix
    percentiles_mat = timeseries.compute_percentiles_matrix(
        np.abs(sttc_mat), np.abs(surrogate_sttc_values)
    )

    # Save both the STTC matrix and the percentiles matrix in a npz file
    # in the session directory. Save also the parameters used to
    # compute the matrices in the same file.
    logging.info("Saving results...")
    np.savez_compressed(
        save_file_path,
        sttc_mat=sttc_mat,
        percentiles_mat=percentiles_mat,
        dt=dt,
        num_trials=num_trials,
        method=method,
        seed=config.seed,
    )


@click.command()
@click.option(
    "--data_dir",
    required=True,
    type=click.Path(dir_okay=True, exists=True),
    help="Path to the dataset directory",
)
@click.option(
    "--n_sessions_in_parallel",
    default=1,
    type=int,
    help="Number of sessions to process in parallel",
)
def main(data_dir, n_sessions_in_parallel=1):
    # Set numpy random seed
    np.random.seed(config.seed)

    # Get list with all sessions in the dataset
    animal_dataset = dataset.AnimalDataset(data_dir)
    sessions_to_loop = utils.get_sessions_to_loop(animal_dataset)

    # Run through all sessions, computing the STTC arrays
    Parallel(n_jobs=n_sessions_in_parallel, verbose=50)(
        delayed(compute_sttc_arrays)(animal_dataset, session)
        for session in sessions_to_loop
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("***** ERROR *****")
        print(traceback.format_exc())
