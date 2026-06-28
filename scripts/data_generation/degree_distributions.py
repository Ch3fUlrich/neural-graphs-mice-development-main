"""
Compiles list of graph degrees for each session, given a choice of similarity
matrix and a connectivity threshold.

Results are saved to a csv file with one row per session. The columns are:
    - animal_id
    - session_date
    - dev_stage
    - degrees: list of degrees for each node in the graph

This script is a thin CLI wrapper around
:func:`ngmd.data_generation.generate_degree_distributions`.
"""

import traceback

import click

from ngmd.data_generation import generate_degree_distributions  # noqa: F401


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
    generate_degree_distributions(data_dir, save_dir, connect_thresh=connect_thresh, sim_type=sim_type)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("***** ERROR *****")
        print(traceback.format_exc())
