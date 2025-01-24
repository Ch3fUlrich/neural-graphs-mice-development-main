"""
Compiles statistics of the binarized traces in the dataset

The results are saved in the save_dir directory as three csv files:
- pulse_rates.gz
- pulse_widths.gz
- pulse_intervals.gz
"""

import os
import traceback

import click
import pandas as pd

from ngmd import (
    dataset,
    timeseries,
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
def main(data_dir, save_dir):
    # Gather animals
    animal_dataset = dataset.AnimalDataset(data_dir)

    # Run through all sessions for all animals
    pulse_rates, pulse_widths, pulse_intervals = compile_pulse_lists(animal_dataset)

    # Concatenate the results
    pulse_rates = pd.concat(pulse_rates, ignore_index=True)
    pulse_widths = pd.concat(pulse_widths, ignore_index=True)
    pulse_intervals = pd.concat(pulse_intervals, ignore_index=True)

    # Make sure the roi column is of type int
    pulse_rates["roi"] = pulse_rates["roi"].astype(int)
    pulse_widths["roi"] = pulse_widths["roi"].astype(int)
    pulse_intervals["roi"] = pulse_intervals["roi"].astype(int)

    # Save the results to file
    pulse_rates.to_csv(
        os.path.join(save_dir, "pulse_rates.gz"), index=False, compression="gzip"
    )
    pulse_widths.to_csv(
        os.path.join(save_dir, "pulse_widths.gz"), index=False, compression="gzip"
    )
    pulse_intervals.to_csv(
        os.path.join(save_dir, "pulse_intervals.gz"), index=False, compression="gzip"
    )

    print("Done. Results saved to {}".format(save_dir))


def compile_pulse_lists(animal_dataset):
    # Initialize the lists of results
    pulse_rates = []
    pulse_widths = []
    pulse_intervals = []

    for animal in animal_dataset.animals:
        print("Processing animal {}".format(animal.animal_id))

        for session in animal.sessions:
            # Load the binarized traces
            binarized_traces = session.load_binarized_traces()

            n_rois, _ = binarized_traces.shape

            # Run through all ROIs
            for roi in range(n_rois):
                # Gather statistics
                info = timeseries.gather_binary_trace_statistics(
                    binarized_traces[roi], session.sample_rate
                )

                # Save the statistics in their respective lists
                pulse_rates.append(
                    pd.DataFrame(
                        {
                            "animal_id": [animal.animal_id],
                            "date": [session.date],
                            "roi": [roi],
                            "n_pulses": [info["n_pulses"]],
                            "duration": [info["duration"]],
                            "pulse_rate": [info["n_pulses"] / info["duration"]],
                            "active_fraction": [info["active_fraction"]],
                            "dev_stage": [session.dev_stage],
                        }
                    )
                )
                pulse_widths.append(
                    pd.DataFrame(
                        {
                            "animal_id": [animal.animal_id] * len(info["pulse_widths"]),
                            "date": [session.date] * len(info["pulse_widths"]),
                            "roi": [roi] * len(info["pulse_widths"]),
                            "pulse_widths": info["pulse_widths"],
                            "dev_stage": [session.dev_stage]
                            * len(info["pulse_widths"]),
                        }
                    )
                )
                pulse_intervals.append(
                    pd.DataFrame(
                        {
                            "animal_id": [animal.animal_id]
                            * len(info["pulse_intervals"]),
                            "date": [session.date] * len(info["pulse_intervals"]),
                            "roi": [roi] * len(info["pulse_intervals"]),
                            "pulse_intervals": info["pulse_intervals"],
                            "dev_stage": [session.dev_stage]
                            * len(info["pulse_intervals"]),
                        }
                    )
                )
    return pulse_rates, pulse_widths, pulse_intervals


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("***** ERROR *****")
        print(traceback.format_exc())
