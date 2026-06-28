"""
Compiles summaries of networks created from the correlation matrices in the
dataset.

Attention: this script currently takes too long to run!

This script is a thin CLI wrapper around
:func:`ngmd.data_generation.generate_small_worldness`.
"""

import traceback

import click

from ngmd.data_generation import generate_small_worldness  # noqa: F401


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
    generate_small_worldness(data_dir, save_dir, sim_type, connect_thresh)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("***** ERROR *****")
        print(traceback.format_exc())
