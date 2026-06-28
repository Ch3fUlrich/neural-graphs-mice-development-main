"""
Computes structural consistency [Lü et al. 2015] for the graph of each
session, given a connectivity threshold.

Results are saved to a csv file with one row per session. The columns are:
    - animal_id
    - session_date
    - dev_stage
    - mean_sc : mean structural consistency (s.c.) across random trials
    - std_sc : standard deviation of s.c. across random trials
    - upper_sc : upper (95th) percentile of s.c. across random trials
    - lower_sc : lower (5th) percentile of s.c. across random trials

This script is a thin CLI wrapper around
:func:`ngmd.data_generation.generate_structural_consistency`.
"""

import traceback

import click

from ngmd.data_generation import generate_structural_consistency  # noqa: F401


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
    generate_structural_consistency(data_dir, save_dir, sim_type, connect_thresh)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("***** ERROR *****")
        print(traceback.format_exc())
