"""
Compiles summaries of networks created from the correlation matrices in the
dataset
"""

import logging
import os
import traceback

import click
import network_summaries_config as config
import networkx as nx
import numpy as np
import pandas as pd
import script_utils
from joblib import Parallel, delayed

from ngmd import (
    dataset,
    graphs,
    utils,
)

# Set up logging
logging.basicConfig(level=logging.INFO)


def initialize_results_df():
    return pd.DataFrame(
        columns=[
            "animal_id",
            "session_date",
            "dev_stage",
            "num_neurons",
            "num_conn_comp",
            "density",
            "transitivity",
            "avg_clust",
            "size_largest_cc",
            "avg_short_path_length",
            "radius",
            "diameter",
            "median_page_rank",
        ]
    )


def _compute_network_metrics(G_nx):
    metrics = {}
    # Number of connected components
    metrics["num_conn_comp"] = nx.number_connected_components(G_nx)
    # Density
    metrics["density"] = nx.density(G_nx)
    # Transitivity (fraction of possible triangles)
    metrics["transitivity"] = nx.transitivity(G_nx)
    # Average clustering coefficient
    metrics["avg_clust"] = nx.algorithms.approximation.average_clustering(G_nx)
    # Relative size of the largest connected component
    largest_cc = G_nx.subgraph(max(nx.connected_components(G_nx), key=len)).copy()
    metrics["size_largest_cc"] = len(largest_cc)
    # Average shortest path length (of the largest connected component)
    metrics["avg_short_path_length"] = nx.average_shortest_path_length(largest_cc)
    # Radius (of the largest connected component)
    metrics["radius"] = nx.radius(largest_cc)
    # Diameter (of the largest connected component)
    metrics["diameter"] = nx.diameter(largest_cc)
    # Median page rank
    metrics["median_page_rank"] = np.median(list(nx.pagerank(G_nx).values()))
    return metrics


def _extract_typical_values(list_of_metrics):
    keys = list_of_metrics[0].keys()
    typical_values = {}
    for key in keys:
        values = [metrics[key] for metrics in list_of_metrics]
        typical_values[key] = np.median(values)
    return typical_values


# Define loop computation
def loop_computation(i, sessions_to_loop, sim_type, connect_thresh):
    # Load similarity matrix
    session = sessions_to_loop[i]
    sim_mat = utils.select_sim_mat(session, sim_type)

    if script_utils.mat_is_valid(sim_mat, config.SUBSAMPLE_SIZE):
        # Build similarity graph
        sg = graphs.SimilarityGraph(sim_mat, connect_thresh)
        # Compute network metrics, possibly over multiple subsampled graphs
        network_metrics = script_utils.subsample_and_compute(
            sg.G_nx,
            _compute_network_metrics,
            config.SUBSAMPLE_SIZE,
            config.NUM_SUBSAMPLE_TRIALS,
        )
        # Extract typical values from the list of metrics
        typical_values = _extract_typical_values(network_metrics)
        # Compile results in a list
        loop_results = [
            session.animal_id,
            session.date,
            session.dev_stage,
            len(sim_mat),  # Number of neurons
        ]
        loop_results.extend([typical_values[key] for key in typical_values.keys()])
        return loop_results
    else:
        return None


def save_to_file(results_df, save_dir, sim_type, connect_thresh):
    file_name = "network_summaries_{}".format(sim_type)
    file_name += "_connect_thresh={:.1f}".format(connect_thresh)
    if config.SUBSAMPLE_SIZE is not None:
        file_name += "_subsample_size={}".format(config.SUBSAMPLE_SIZE)
    file_name += ".csv"
    file_path = os.path.join(save_dir, file_name)
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
    help="Type of similarity matrix to use. Default is 'sttc_percentiles'",
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
    results_df = initialize_results_df()

    # Make list of sessions to go through
    sessions_to_loop = utils.get_sessions_to_loop(animal_dataset)

    # Run loop computation
    results_list = Parallel(n_jobs=-1, verbose=50)(
        delayed(loop_computation)(i, sessions_to_loop, sim_type, connect_thresh)
        for i in range(len(sessions_to_loop))
    )

    # Add results to data frame
    for row in results_list:
        if row is not None:
            results_df.loc[len(results_df)] = row

    # Save results to csv file
    save_to_file(results_df, save_dir, sim_type, connect_thresh)
    logging.info("Done! Results saved to {}".format(save_dir))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("***** ERROR *****")
        print(traceback.format_exc())
