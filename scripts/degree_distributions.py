"""
Compiles list of graph degrees for each session, given a choice of similarity
matrix and a connectivity threshold

The results are saved to a csv file with one row per session. The columns are:
    - animal_id
    - session_date
    - dev_stage
    - degrees: list of degrees for each node in the graph, computed from the
    binary adjacency matrix

Usage:
    python scripts/degree_distributions.py \
        --data_dir <path_to_data_dir> \
        --save_dir <path_to_save_dir> \
        --sim_type <similarity_type> \
        --connect_thresh <connectivity_threshold>

"""

import os
import traceback

import click
import pandas as pd
from tqdm import tqdm

from ngmd import (
    dataset,
    graphs,
    utils,
)


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
    help="Connectivity threshold for creating graphs from " + "similarity matrices",
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
            "degrees",
        ]
    )

    # Run through all sessions for all animals
    for animal in animal_dataset.animals:
        print("Processing animal {}".format(animal.animal_id))
        for session in tqdm(animal.sessions):
            # Load similarity matrix
            sim_mat = utils.select_sim_mat(session, sim_type)

            if sim_mat is not None:
                # Build similarity graph
                sg = graphs.SimilarityGraph(
                    similarity_matrix=sim_mat, connect_thresh=connect_thresh
                )

                # Get degrees and weighted degrees from the networkx graph. Save to
                # list of floats to help when loading these values from file later
                degrees = [float(d) for (_, d) in sg.G_nx.degree()]

                # Append to statistics to results data frame
                new_row = [
                    session.animal_id,
                    session.date,
                    session.dev_stage,
                    degrees,
                ]
                results.loc[len(results)] = new_row

    # Save results to csv file
    file_path = os.path.join(
        save_dir,
        "degree_distributions_{}_".format(sim_type)
        + "connect_thresh={:.1f}.csv".format(connect_thresh),
    )
    results.to_csv(file_path, index=False)

    return


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("***** ERROR *****")
        print(traceback.format_exc())
