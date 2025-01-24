"""
Computes structural consistency [Lü et al. 2015] for the graph of each
session, given a connectivity threshold

The results are saved to a csv file with one row per session. The columns are:
    - animal_id
    - session_date
    - dev_stage
    - mean_sc : mean structural consistency (s.c.) across random trials
    - std_sc : standard deviation of s.c. across random trials
    - upper_sc : upper (95th) percentile of s.c. across random trials
    - lower_sc : lower (5th) percentile of s.c. across random trials

Usage:
    python scripts/structural_consistency.py \
        --data_dir <path_to_data_dir> \
        --save_dir <path_to_save_dir> \
        --sim_type <similarity_type> \
        --connect_thresh <connectivity_threshold>
        --n_trials <number_of_random_trials>

"""

import logging
import os
import traceback
import warnings

import click
import numpy as np
import pandas as pd
import script_utils
import structural_consistency_config as config
from joblib import Parallel, delayed
from tqdm import tqdm

from ngmd import (
    dataset,
    graphs,
    utils,
)
from ngmd import structural_consistency as sc

# Set up logging
logging.basicConfig(level=logging.INFO)


def _compile_statistics(trial_scs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.mean(trial_scs)
        std = np.std(trial_scs)
        upper = np.percentile(trial_scs, 95)
        lower = np.percentile(trial_scs, 5)
    return mean, std, upper, lower


@click.command()
@click.option(
    "--data_dir",
    required=True,
    type=click.Path(dir_okay=True, exists=True),
    help="Path to the dataset directory",
)
@click.option(
    "--save_dir",
    required=True,
    type=click.Path(dir_okay=True, exists=True),
    help="Path to the directory where to save the output",
)
@click.option(
    "--sim_type",
    required=False,
    type=click.Choice(["zscores", "sttc_percentiles"]),
    default="sttc_percentiles",
    help="Type of similarity matrix to use. Default is sttc_percentiles",
)
@click.option(
    "--connect_thresh",
    required=True,
    type=click.FLOAT,
    help="Connectivity threshold for creating graphs from similarity matrices",
)
def main(data_dir, save_dir, sim_type, connect_thresh):
    # Gather animals
    animal_dataset = dataset.AnimalDataset(data_dir)

    # Create results data frame
    results = pd.DataFrame(
        columns=[
            "animal_id",
            "session_date",
            "dev_stage",
            "mean_sc",
            "std_sc",
            "upper_sc",
            "lower_sc",
        ]
    )

    # Run through all sessions for all animals
    for animal in animal_dataset.animals:
        print("Processing animal {}".format(animal.animal_id))
        for session in tqdm(animal.sessions):
            # Load similarity matrix
            sim_mat = utils.select_sim_mat(session, sim_type)

            if script_utils.mat_is_valid(sim_mat, config.SUBSAMPLE_SIZE):
                # Build similarity graph
                sg = graphs.SimilarityGraph(
                    similarity_matrix=sim_mat, connect_thresh=connect_thresh
                )

                # Define trial function for computing structural consistency
                def trial_fun():
                    return script_utils.subsample_and_compute(
                        sg.G_nx,
                        sc.structural_consistency,
                        config.SUBSAMPLE_SIZE,
                        config.NUM_SUBSAMPLE_TRIALS,
                    )

                # Compute structural consistency
                trial_scs = Parallel(n_jobs=-1, verbose=1)(
                    delayed(trial_fun)() for _ in range(config.NUM_TRIALS)
                )

                # Flatten list of lists
                trial_scs = [sc for sublist in trial_scs for sc in sublist]

                # Compute statistics
                mean, std, upper, lower = _compile_statistics(trial_scs)

                # Append to statistics to results data frame
                new_row = [
                    session.animal_id,
                    session.date,
                    session.dev_stage,
                    mean,
                    std,
                    upper,
                    lower,
                ]
                results.loc[len(results)] = new_row

    # Save results to csv file
    file_name = "structural_consistency_{}".format(sim_type)
    file_name += "_connect_thresh={:.1f}".format(connect_thresh)
    if config.SUBSAMPLE_SIZE is not None:
        file_name += "_subsample_size={}".format(config.SUBSAMPLE_SIZE)
    file_name += ".csv"
    file_path = os.path.join(save_dir, file_name)
    results.to_csv(file_path, index=False)

    return


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("***** ERROR *****")
        print(traceback.format_exc())
