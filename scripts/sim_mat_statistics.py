"""
Compiles statistics of the similarity matrices in the dataset

The result is saved as a table with one row per experiment session in the
dataset. Each row is identified by an animal ID, a session date, and a
development stage marker. The rows also contain counts and bin edges for the
(non-trivial) similarity values found in the session data, along with maximum,
minimum, average and standard deviation of said values.
"""

import os
import traceback

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from ngmd import (
    dataset,
    utils,
)


def compute_histogram(values, num_bins):
    counts, bins = np.histogram(
        values,
        bins=num_bins,
        range=(-1.0, 1.0),
    )
    # Save to list of floats to help when loading these values from
    # file later
    counts = counts.astype(float).tolist()
    bins = bins.astype(float).tolist()
    return counts, bins


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
    "--num_bins",
    required=False,
    type=click.INT,
    default=100,
    help="Number of bins to use when computing the "
    + "histograms of similarity values. Default is 100.",
)
@click.option(
    "--sim_type",
    required=False,
    type=click.Choice(["corrs", "sttc"]),
    default="sttc",
    help="Type of similarity matrix to use. Default is " + "sttc",
)
def main(data_dir, save_dir, num_bins, sim_type):
    # Gather animals
    animal_dataset = dataset.AnimalDataset(data_dir)

    # Create results data frame
    results = pd.DataFrame(
        columns=[
            "animal_id",
            "session_date",
            "dev_stage",
            "num_neurons",
            "sim_counts",
            "sim_bins",
            "min_sim_val",
            "max_sim_val",
            "avg_sim_val",
            "std_sim_val",
        ]
    )

    # Run through all sessions for all animals
    for animal in animal_dataset.animals:
        print("Processing animal {}".format(animal.animal_id))
        for session in tqdm(animal.sessions):
            # Load similarity matrix
            sim_mat = utils.select_sim_mat(session, sim_type)

            if sim_mat is not None:
                # Get non-trivial values
                non_trivial_sim_vals, num_neurons = utils.get_non_trivial_values(
                    sim_mat
                )

                # Compute histogram of values
                counts, bins = compute_histogram(non_trivial_sim_vals, num_bins)

                # Append to statistics to results data frame
                new_row = [
                    session.animal_id,
                    session.date,
                    session.dev_stage,
                    num_neurons,
                    counts,
                    bins,
                    np.min(non_trivial_sim_vals),
                    np.max(non_trivial_sim_vals),
                    np.mean(non_trivial_sim_vals).astype("float"),
                    np.std(non_trivial_sim_vals).astype("float"),
                ]
                results.loc[len(results)] = new_row

    # Save results to csv file
    results.to_csv(
        os.path.join(save_dir, "{}_statistics.csv".format(sim_type)), index=False
    )

    return


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("***** ERROR *****")
        print(traceback.format_exc())
