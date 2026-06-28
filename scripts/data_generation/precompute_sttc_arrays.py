"""
Precompute STTC arrays for all sessions in the dataset.

Important: Make sure to set up the ``precompute_sttc_arrays`` section in
``configs/config.yaml`` before running this script.

This script is a thin CLI wrapper around
:func:`ngmd.data_generation.generate_sttc_arrays`.
"""

import traceback

import click

from ngmd.data_generation import compute_sttc_arrays, generate_sttc_arrays  # noqa: F401


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
    generate_sttc_arrays(data_dir, n_sessions_in_parallel=n_sessions_in_parallel)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("***** ERROR *****")
        print(traceback.format_exc())
