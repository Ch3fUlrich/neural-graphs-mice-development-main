"""
Compiles statistics of the binarized traces in the dataset.

Results are saved in ``save_dir`` as:
- ``pulse_rates.gz``
- ``pulse_widths.gz``
- ``pulse_intervals.gz``

This script is a thin CLI wrapper around
:func:`ngmd.data_generation.generate_binarized_traces_statistics`.
"""

import traceback

import click

from ngmd.data_generation import generate_binarized_traces_statistics  # noqa: F401


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
def main(data_dir, save_dir):
    generate_binarized_traces_statistics(data_dir, save_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("***** ERROR *****")
        print(traceback.format_exc())
