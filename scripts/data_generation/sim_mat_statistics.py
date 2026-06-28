"""
Compiles statistics of the similarity matrices in the dataset.

The result is saved as a table with one row per experiment session. Each row
is identified by an animal ID, a session date, and a development stage marker.
The rows also contain counts and bin edges for the (non-trivial) similarity
values found in the session data, along with max, min, average and std of
said values.

This script is a thin CLI wrapper around
:func:`ngmd.data_generation.generate_sim_mat_statistics`.
"""

import traceback

import click

from ngmd.data_generation import generate_sim_mat_statistics  # noqa: F401


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
    help="Number of bins to use when computing the histograms of similarity values. Default is 100.",
)
@click.option(
    "--sim_type",
    required=False,
    type=click.Choice(["corrs", "sttc"]),
    default="sttc",
    help="Type of similarity matrix to use. Default is sttc",
)
def main(data_dir, save_dir, num_bins, sim_type):
    generate_sim_mat_statistics(data_dir, save_dir, num_bins=num_bins, sim_type=sim_type)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("***** ERROR *****")
        print(traceback.format_exc())
