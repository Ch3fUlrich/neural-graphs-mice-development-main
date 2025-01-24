"""Utilities for the notebooks"""

import networkx as nx
import numpy as np
import pandas as pd

from ngmd import graphs, utils


def labels_from_cuts(cuts):
    """Get list of labels corresponding to the specified development stage cuts

    Parameters
    ----------
    cuts : list of int
        List of  development stage cuts in the units of PDay

    Returns
    -------
    list of str
        List of development window labels in the format PDay x-y, where x and y
        are cuts

    Examples
    --------
    >>> labels_from_cuts([0, 15, 25, 45, 65, 85])
    ['PDay 0-15', 'PDay 15-25', 'PDay 25-45', 'PDay 45-65', 'PDay 65-85']
    """
    labels = []
    for i in range(len(cuts) - 1):
        labels.append("PDay {}-{}".format(int(cuts[i]), int(cuts[i + 1])))
    return labels


def dev_stage_to_label(dev_stages, dev_stage_cuts):
    """Convert development stages to labels

    Parameters
    ----------
    dev_stages : array-like
        Array of development stages in the units of PDay
    dev_stage_cuts : list of int
        List of development stage cuts in the units of PDay

    Returns
    -------
    list of str
        List of development window labels in the format PDay x-y, where x and y
        are cuts

    Examples
    --------
    >>> dev_stage_to_label([15, 90, 24, 32, 40], [0, 15, 25, 45, 65, 85])
    ['PDay 0-15', 'PDay 65-85', 'PDay 15-25', 'PDay 25-45', 'PDay 25-45']
    """
    dev_stage_cuts = np.array(dev_stage_cuts)
    dev_stage_labels = labels_from_cuts(dev_stage_cuts)
    result = []
    for dev_stage in dev_stages:
        i = np.clip(np.sum(dev_stage > dev_stage_cuts), 0, len(dev_stage_labels) - 1)
        result.append(dev_stage_labels[i])
    return result


def group_by_dev_interval(df, bins, labels=None):
    """Group data frame by development stage windows

    Parameters
    ----------
    df : `pandas.DataFrame`
        Input data frame. Must have a column named `'dev_stage'` containing a
        PDay value for each row
    bins : list of int
        List of development stage cuts from which to define the development
        stage windows. Must have length equal to the number of desired windows
        plus one
    labels : list of str, optional
        List of label names to assign to the development windows, by default
        the function `labels_from_cuts` is run. Must have length equal to the
        number of desired windows.

    Returns
    -------
    `pandas.DataFrameGroupBy`
        A groupby object that contains information about the groups
    """

    if labels is None:
        labels = labels_from_cuts(bins)

    dev_interval_grouping = df.groupby(
        pd.cut(df["dev_stage"], bins=bins, labels=labels),
        observed=True,
    )

    return dev_interval_grouping


def get_prob_masses(samples, xrange, n_bins=100, return_bins=False):
    """Generate empirical probability masses from the histogram of a sample

    Parameters
    ----------
    samples : array-like
        Samples
    xrange : tuple
        Lower and upper bound on the values that the samples can take
    n_bins : int, optional
        Number of bins when computing the samples' histogram, by default 100
    return_bins : bool, optional
        Return bins in addition to the histogram values, by default False

    Returns
    -------
    array-like
        List of probability masses, one for each bin in the histogram
    """
    hist, bins = np.histogram(samples, bins=n_bins, range=xrange, density=False)

    prob_masses = hist.astype(float) / np.sum(hist)

    if return_bins:
        return prob_masses, bins
    else:
        return prob_masses


def get_typical_samples(summaries_df, summary_name, dev_window, num_samples=3):
    """Get samples with typical value from a summary column of a data frame

    Parameters
    ----------
    summaries_df : `pandas.DataFrame`
        The input data frame. Must have columns `'connect_thresh'`,
        `'dev_stage'`, `'animal_id'`, `'session_date'`, and `summary_name`
    summary_name : str
        Name of the summary column of interest
    dev_window : tuple
        Development window of interest
    num_samples : int, optional
        Number of samples to return, by default 3

    Returns
    -------
    list of dict
        List of dictionaries containing the `'animal_id'`, `'session_date'`,
        and summary value for each of the selected samples
    """
    # Restrict summaries to the developmental window of interest
    dev_grouping = group_by_dev_interval(summaries_df, dev_window)
    summaries_df = dev_grouping.get_group(list(dev_grouping.groups.keys())[0])

    # Get the median (typical) value of the summary of interest
    typical_value = np.median(summaries_df[summary_name])

    # Get the samples (rows) who's selected summary is closest to the typical
    # value
    typical_samples = summaries_df.iloc[
        (summaries_df[summary_name] - typical_value).abs().argsort()[:num_samples]
    ]

    # Build list of dictionaries containing the `'animal_id'`,
    # `'session_date'`, and summary value for each of the selected samples
    results = []
    for _, row in typical_samples.iterrows():
        results.append(
            {
                "animal_id": row["animal_id"],
                "session_date": row["session_date"],
                summary_name: row[summary_name],
            }
        )

    return results


def build_networks(animal_dataset, df, sim_type, connect_thresh):
    """Build similarity networks from a data frame

    Parameters
    ----------
    animal_dataset : `ngmd.dataset.AnimalDataset`
        The input animal dataset
    df : `pandas.DataFrame`
        The input data frame. Must have columns `'animal_id'` and
        `'session_date'`. The values in these columns specify the data from
        which to build the graphs
    sim_type : str
        Type of similarity values to use when building the graphs. See
        `ngmd.utils.select_sim_mat` for options.
    connect_thresh : float
        Connectivity threshold for the similarity networks

    Returns
    -------
    list of dict
        List of dictionaries containing the `'animal_id'`, `'session_date'`,
        and `'graph'` for each of the selected samples
    """
    networks = []
    for row in df:
        animal = animal_dataset.get_animal_from_id(row["animal_id"])
        session = animal.get_session_from_date(str(row["session_date"]))
        sim_mat = utils.select_sim_mat(session, sim_type)
        networks.append(
            {
                "animal_id": row["animal_id"],
                "session_date": row["session_date"],
                "graph": graphs.SimilarityGraph(sim_mat, connect_thresh=connect_thresh),
            }
        )
    return networks


def get_cc_sizes(animal_dataset, summaries_df, sim_type, normalized=False):
    # Run through each animal/session and get a list of the number of connected
    # components in its graph
    cc_sizes = []
    for animal_id, session_date in zip(
        summaries_df["animal_id"], summaries_df["session_date"]
    ):
        animal = animal_dataset.get_animal_from_id(animal_id)
        session = animal.get_session_from_date(str(session_date))
        sim_mat = utils.select_sim_mat(session, sim_type)
        graph = graphs.SimilarityGraph(
            sim_mat, connect_thresh=summaries_df["connect_thresh"]
        )
        # Run through each connected component and add its size to the list
        for component in nx.connected_components(graph.G_nx):
            if normalized:
                cc_sizes.append(len(component) / len(graph.G_nx))
            else:
                cc_sizes.append(len(component))

    return cc_sizes


def select_default_connect_thresh(sim_method):
    if sim_method == "corrs":
        return 0.2
    elif sim_method == "zscores":
        return 4
    elif sim_method == "sttc":
        return 0.2
    elif sim_method == "sttc_percentiles":
        return 95
    else:
        return


def update_bad_sessions(df, animal_id, dev_stage, animal_dataset):
    """Update the bad sessions table in the animal dataset

    Parameters
    ----------
    df : `pandas.DataFrame`
        The input data frame. Must have columns `'animal_id'`, `'dev_stage'`,
        and `'session_date'`.
    animal_id : str
        The animal id of the bad session
    dev_stage : int
        The development stage of the bad session
    animal_dataset : `ngmd.dataset.AnimalDataset`
        The animal dataset object used to update the bad sessions table
    """
    # Select bad session row from animal_id and dev_stage
    bad_session_row = df[
        (df["animal_id"] == animal_id) & (df["dev_stage"] == dev_stage)
    ]
    # Keep only the columns needed for the record
    bad_measurements_df = bad_session_row[["animal_id", "session_date"]]
    # Rename the column session_date to date
    bad_measurements_df = bad_measurements_df.rename(columns={"session_date": "date"})
    # Update the bad sessions table in the animal dataset
    animal_dataset.update_bad_measurements(bad_measurements_df)


def keep_rows_from_good_sessions(df, animal_dataset):
    """Keep rows from good sessions

    Parameters
    ----------
    df : `pandas.DataFrame`
        The input data frame. Must have columns `'animal_id'` and
        `'session_date'`. The values in these columns specify the data from
        which to build the graphs
    animal_dataset : `ngmd.dataset.AnimalDataset`
        The input animal dataset

    Returns
    -------
    `pandas.DataFrame`
        The input data frame with only the rows that correspond to good
        sessions
    """
    # Load bad sessions data frame
    bad_sessions = animal_dataset.load_bad_measurements_table(which="sessions")
    # Turn it into a list of tuples
    bad_sessions = bad_sessions.to_records(index=False).tolist()
    # Filter out rows that correspond to bad sessions
    selected_df = df[
        df.apply(
            lambda row: (row["animal_id"], row["session_date"]) not in bad_sessions,
            axis=1,
        )
    ]
    return selected_df


def format_table_of_passed_tests(results):
    """Format table of pairwise tests that passed the null hypothesis

    Parameters
    ----------
    results : dict
        Dictionary containing the results of the pairwise tests. Must have
        keys 'label_pairs', 'reject_null', 'test_stats', and 'p_vals'

    Returns
    -------
    `pandas.DataFrame`
        Data frame containing the test statistics and p-values of the tests
        that passed the null hypothesis
    """
    dfs = []
    for i in range(len(results["label_pairs"])):
        if results["reject_null"][i]:
            dfs.append(
                pd.DataFrame(
                    {
                        "pair": results["label_pairs"][i][0]
                        + " vs. "
                        + results["label_pairs"][i][1],
                        "test_stat": results["test_stats"][i],
                        "p_val": results["p_vals"][i],
                    },
                    index=[0],
                )
            )
    try:
        return pd.concat(dfs, ignore_index=True)
    except ValueError:
        return pd.DataFrame(columns=["pair", "test_stat", "p_val"])
