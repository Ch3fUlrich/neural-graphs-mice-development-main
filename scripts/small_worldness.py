"""
Compiles summaries of networks created from the correlation matrices in the
dataset

Attention: this script currently takes too long to run!
"""

import logging
import os
import traceback

import click
import networkx as nx
import numpy as np
import pandas as pd

from ngmd import (
    dataset,
    graphs,
    utils,
)

# Set up logging
logging.basicConfig(level=logging.INFO)


def loop_computation(i, sessions_to_loop, **kwargs):
    # Parse keyword arguments
    sim_type = kwargs.pop("sim_type")
    connect_thresh = kwargs.pop("connect_thresh")

    # Load similarity matrix
    sim_mat = utils.select_sim_mat(sessions_to_loop[i], sim_type)

    if sim_mat is not None:
        # Build similarity graph
        sg = graphs.SimilarityGraph(
            similarity_matrix=sim_mat, connect_thresh=connect_thresh
        )

        # Relative size of the largest connected component
        largest_cc = graphs.largest_cc(sg.G_nx)

        # Small worldness: use omega because [Neal 2017] recommends
        # not using sigma
        try:
            small_worldness = graphs.omega(
                largest_cc, nrand=5, n_jobs=-1, verbose=50, timeout=180
            )
        except nx.exception.NetworkXError:
            # This error can be raised, for instance, when the
            # graph has less than four nodes. In this case, we also
            # assign np.nan to omega.
            small_worldness = np.nan

        # Compile results in a list
        loop_results = [
            sessions_to_loop[i].animal_id,
            sessions_to_loop[i].date,
            sessions_to_loop[i].dev_stage,
            small_worldness,
        ]

        return loop_results

    else:
        return None


def save_to_file(results_df, save_dir, sim_type, connect_thresh):
    file_path = os.path.join(
        save_dir,
        "small_worldness_{}_".format(sim_type)
        + "connect_thresh={:.1f}.csv".format(connect_thresh),
    )
    results_df.to_csv(file_path, index=False)


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
    help="Connectivity threshold for creating graphs from " + "similarity matrices",
)
def main(data_dir, save_dir, sim_type, connect_thresh):
    # Make list of sessions to go through
    animal_dataset = dataset.AnimalDataset(data_dir)
    sessions_to_loop = utils.get_sessions_to_loop(animal_dataset)

    # Create results data frame
    results_df = pd.DataFrame(
        columns=[
            "animal_id",
            "session_date",
            "dev_stage",
            "small_worldness",
        ]
    )

    # Run loop computation series. It's faster do do this is series and compute omega
    # in parallel
    results_list = [
        loop_computation(
            i, sessions_to_loop, sim_type=sim_type, connect_thresh=connect_thresh
        )
        for i in range(len(sessions_to_loop))
    ]

    # Add results to data frame
    for row in results_list:
        if row is not None:
            results_df.loc[len(results_df)] = row

    # Save results to csv file
    save_to_file(results_df, save_dir, sim_type, connect_thresh)

    return


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("***** ERROR *****")
        print(traceback.format_exc())
