"""Utilities module"""

import numpy as np
import pandas as pd


class set_string:
    """Set string format when printing

    Notes
    -----
    Always call `set_string.END` at the end of the print statement to return
    subsequent print statements to the default state
    """

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def keep_strings_ending_in(string_list, keys=[]):
    """Keep only strings ending in a given set of keys

    Parameters
    ----------
    string_list : list of str
        List of strings
    keys : list of str, optional
        List of keys to keep, by default []

    Returns
    -------
    list of str
        List of strings ending in any of the keys in `keys`
    """
    pruned_list = [s for s in string_list if any(s.endswith(key) for key in keys)]
    return pruned_list


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
    """
    labels = []
    for i in range(len(cuts) - 1):
        labels.append("PDay {}-{}".format(int(cuts[i]), int(cuts[i + 1])))
    return labels


def select_sim_mat(session, sim_type):
    """Select similarity matrix from a session

    Parameters
    ----------
    session : Session
        Session object
    sim_type : str
        Type of similarity matrix to select

    Returns
    -------
    array-like
        Similarity matrix
    """
    if sim_type == "corrs":
        sim_mat = session.load_corrs()
    elif sim_type == "zscores":
        sim_mat = session.load_zscores()
    elif sim_type == "sttc":
        sim_mat = session.load_sttc_array()
    elif sim_type == "sttc_percentiles":
        sim_mat = session.load_sttc_percentiles()
    else:
        raise ValueError("Invalid similarity type")
    return sim_mat


def fill_sym_mat_from_triu(n, triu_values, diag=0):
    """Fill a symmetric matrix from its upper triangular part

    Parameters
    ----------
    n : int
        Number of rows and columns in the matrix
    triu_values : array-like
        Upper triangular part of the matrix
    diag : float, optional
        Diagonal values, by default 0

    Returns
    -------
    array-like
        Symmetric matrix
    """
    # Initialize symmetric matrix
    sym_mat = np.zeros((n, n))

    # Fill upper triangular part
    sym_mat[np.triu_indices(n, 1)] = triu_values

    # Fill lower triangular part
    sym_mat += sym_mat.T

    # Fill diagonal
    np.fill_diagonal(sym_mat, diag)

    return sym_mat


def update_table(new_rows_df, column_names, save_path):
    """Update a table with new rows

    Parameters
    ----------
    new_rows_df : pandas.DataFrame


    Notes
    -----
    The table of bad ROIs is saved in a CSV file named 'bad_rois.csv' in
    the root directory of the dataset. If the file does not exist, it is
    created. If the file already exists, the new bad ROIs are appended to
    the existing table, and duplicates are removed.
    """
    # Try to load the existing outliers from the file
    try:
        existing_table = pd.read_csv(
            save_path,
            dtype={"animal_id": str, "date": str},
        )
    except FileNotFoundError:
        existing_table = pd.DataFrame(columns=column_names)

    # Concatenate the existing and new outliers
    updated_table = pd.concat([existing_table, new_rows_df.copy()])
    if "animal_id" in updated_table:
        updated_table["animal_id"] = updated_table["animal_id"].astype(str)
    if "date" in updated_table:
        updated_table["date"] = updated_table["date"].astype(str)

    # Drop duplicates
    updated_table = updated_table.drop_duplicates()

    # Save the updated outliers to the file
    updated_table.to_csv(save_path, index=False)


def get_non_trivial_values(sim_mat):
    """Get non-trivial values from a similarity matrix

    The list of non-trivial values comprises all off-diagonal values of the
    similarity matrix, after filtering for NaN values

    Parameters
    ----------
    sim_mat : array-like
        Similarity matrix

    Returns
    -------
    array-like
        Non-trivial values
    int
        Number of neurons
    """
    num_neurons = len(sim_mat)
    # The diagonal of the similarity matrix (self-similarity)
    # contains only trivial values
    mask = ~np.eye(num_neurons).astype(bool)
    non_trivial_sim_vals = sim_mat[mask].flatten()

    # The matrix may still contain NaN values. Remove them
    non_trivial_sim_vals = non_trivial_sim_vals[~np.isnan(non_trivial_sim_vals)]
    return non_trivial_sim_vals, num_neurons


def get_sessions_to_loop(animal_dataset):
    sessions_to_loop = []
    for animal in animal_dataset.animals:
        for session in animal.sessions:
            sessions_to_loop.append(session)

    return sessions_to_loop


def get_default_dev_stage_cuts():
    """Get default development stage cuts

    Loads development stage cuts from the project configuration file.

    Returns
    -------
    list of int
        List of default development stage cuts
    """
    from ngmd.config import get_config_section
    
    config = get_config_section("development_stages", defaults={"cuts": [15, 24, 55, 90, 300]})
    return config["cuts"]
