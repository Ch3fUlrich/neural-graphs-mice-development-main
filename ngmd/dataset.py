"""Dataset processing module"""

import json
import os
import warnings
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import yaml

from ngmd.utils import update_table


def get_animal_paths(data_dir):
    """List the paths of all subdirectories containing animal data

    Parameters
    ----------
    data_dir : str
        Path to the directory where to root the search

    Returns
    -------
    list
        List of paths of animal data
    """
    # Get paths to the year directories with pattern '20??'
    year_paths = glob(os.path.join(data_dir, "20??"))

    # For each year, get paths to all directories following the pattern
    # 'DON-*'. These are the animal root directories
    animal_paths = []
    for year_path in year_paths:
        animal_paths += glob(os.path.join(year_path, "DON-*"))

    return animal_paths


def load_dev_stages(json_path):
    """Load developmental stages information

    Parameters
    ----------
    json_path : str
        Path to the `dev_stages.json` file

    Returns
    -------
    dict
        Hierarchical dictionary with developmental stage information. The first
        set of keys correspond to animal IDs. At the next level, to each animal
        ID correspond some experiment session dates. To each experiment session
        date, a label is attached, indicating what developmental stage that
        animal found itself in at that session. An empty dictionary is returned
        if the file indicated by `json_path` does not exist
    """
    try:
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        json_data = {}
    return json_data


class AnimalDataset:
    """Animal dataset class

    Parameters
    ----------
    path : str
        Path to the root of the dataset

    Attributes
    ----------
    path : str
        Path to the root of the dataset
    animal_paths : list
        List of paths to the animal data
    animal_list : list of `ngmd.dataset.Animal`
        `Animal` objects corresponding to each of the animals in the dataset
    bad_rois : pandas.DataFrame
        DataFrame containing the bad ROIs. If the table does not exist, an empty
        DataFrame is returned
    bad_sessions : pandas.DataFrame
        DataFrame containing the bad sessions. If the table does not exist, an
        empty DataFrame is returned

    Notes
    -----
    The dataset is supposed to be organized as follows. The root directory
    contains subdirectories named `'DON-??????'`, indicating the animal ID.
    Each such subdirectory contains a yaml metadata file with the same name as
    its parent directory. The session data is contained in subdirectories
    following the naming convention `'YYYYMMDD'`, representing different
    measurement days for the respective `'DON-??????'` animal.
    """

    def __init__(self, path) -> None:
        self.path = path
        self.animals_info = self._read_metadata_files()
        self.animals = self._load_animals()
        self.bad_rois = self.load_bad_measurements_table(which="rois")
        self.bad_sessions = self.load_bad_measurements_table(which="sessions")

    def _read_metadata_files(self):
        # List all yaml files in `self.path` that follow the pattern 'DON-*'
        # and read them
        animal_info_files = glob(os.path.join(self.path, "DON-*", "DON-*.yaml"))
        animals_info = []
        for animal_info_file in animal_info_files:
            with open(animal_info_file, "r") as yaml_file:
                # Load the yaml file into a temporary dictionary
                tmp = yaml.safe_load(yaml_file)
                # Restructure the temporary dictionary so that the new entries
                # match the arguments of the Animal class
                animal_info_dict = {
                    "cohort_year": tmp["cohort_year"],
                    "animal_id": tmp["animal_id"],
                    "dob": tmp["dob"],
                    "implanted": tmp["implanted"],
                    "injected": tmp["injected"],
                    "sex": tmp["sex"],
                }
                animals_info.append(animal_info_dict)
        return animals_info

    def _load_animals(self):
        animals = []
        for entry in self.animals_info:
            kwargs = {
                "path": os.path.join(self.path, entry["animal_id"]),
                "animal_id": entry["animal_id"],
                "dob": entry["dob"],
            }
            animals.append(Animal(**kwargs))
        return animals

    def get_animal_from_id(self, animal_id):
        """Get animal object from animal ID

        Parameters
        ----------
        animal_id : str
            Animal ID, following the pattern `'DON-??????'`

        Returns
        -------
        `ngmd.dataset.Animal`
            Animal object
        """
        out_animal = None
        for animal in self.animals:
            if animal.animal_id == animal_id:
                out_animal = animal
                break
        return out_animal

    def get_animals_from_year(self, year):
        """Get animal objects from year

        Parameters
        ----------
        year : str
            Year of the experiment

        Returns
        -------
        list of `ngmd.dataset.Animal`
            Animal objects
        """
        out_animals = []
        for animal in self.animals:
            if animal.year == year:
                out_animals.append(animal)
        return out_animals

    def get_sessions_from_dev_stage_range(self, dev_stage_range: tuple):
        """Get sessions from developmental stage range

        Parameters
        ----------
        dev_stage_range : tuple
            Tuple of two integers indicating the lower and upper bounds of the
            developmental stage range

        Returns
        -------
        list of `ngmd.dataset.Session`
            `Session` objects
        """
        out_sessions = []
        for animal in self.animals:
            for session in animal.sessions:
                if session.dev_stage is not None:
                    if (
                        session.dev_stage >= dev_stage_range[0]
                        and session.dev_stage <= dev_stage_range[1]
                    ):
                        out_sessions.append(session)
        return out_sessions

    def update_bad_measurements(self, bad_measurements_df):
        """Update the table of bad ROIs

        Parameters
        ----------
        bad_measurements_df : pandas.DataFrame
            DataFrame containing the bad measurements. Must have columns
            'animal_id', and 'date'. If the DataFrame has a column
            'roi', the table of bad ROIs is updated. If the DataFrame has no
            'roi' column, the table of bad sessions is updated.

        See Also
        --------
        ngmd.utils.update_table
        """
        column_names = bad_measurements_df.columns
        if "animal_id" not in column_names:
            raise ValueError("Column 'animal_id' is missing")
        if "date" not in column_names:
            raise ValueError("Column 'date' is missing")

        if "roi" in column_names:
            file_name = "bad_rois.csv"
        else:
            file_name = "bad_sessions.csv"

        update_table(
            bad_measurements_df,
            column_names=column_names,
            save_path=os.path.join(self.path, file_name),
        )

    def load_bad_measurements_table(self, which="rois"):
        """Load the table of bad ROIs or bad sessions

        Parameters
        ----------
        which : str, optional
            Which table to load, either 'rois' or 'sessions', by default 'rois'

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the bad measurements. If the table does not
            exist, an empty DataFrame is returned

        Raises
        ------
        ValueError
            If `which` is neither 'rois' nor 'sessions'
        """
        if which not in ["rois", "sessions"]:
            raise ValueError("which must be either 'rois' or 'sessions'")
        try:
            bad_measurements = pd.read_csv(
                os.path.join(self.path, "bad_{}.csv".format(which))
            )
        except FileNotFoundError:
            bad_measurements = pd.DataFrame()
        return bad_measurements

    def load_traces_from_good_rois(self, session):
        """Load binarized traces from the good ROIs in the provided session

        Parameters
        ----------
        session : ngmd.dataset.Session
            Session object

        Returns
        -------
        array_like
            An N-by-T array of calcium fluorescence traces, where N is the number
            of ROIs and T is the number of time points in the recording. If the
            traces do not exist, returns None
        """
        # Load the binarized traces
        binarized_traces = session.load_binarized_traces(clean=True)

        # Get rows from self.bad_rois that correspond to the provided session
        bad_rois = self.bad_rois.loc[
            (self.bad_rois["animal_id"] == session.animal_id)
            & (self.bad_rois["date"] == session.date)
        ]["roi"].tolist()

        # Remove the bad ROIs from the binarized traces
        good_rois = np.setdiff1d(np.arange(binarized_traces.shape[0]), bad_rois)
        good_traces = binarized_traces[good_rois, :]

        return good_traces


class Animal:
    """Animal class

    Parameters
    ----------
    path : str
        Path to the directory `'<animal_id>/'` containing
        the animal files
    animal_id : str
        Identification number of the animal used in the experiment session
    dob : str
        Date of birth of the animal, in the format 'YYYYMMDD'

    Attributes
    ----------
    path : str
        Path to the directory `'<animal_id>/'` containing
        the animal files
    animal_id : str
        Identification number of the animal used in the experiment session
    dob : str
        Date of birth of the animal, in the format 'YYYYMMDD'
    sessions : list of `ngmd.dataset.Session`
        `Session` objects corresponding to each of the experiment sessions that
        this animal took place in

    Methods
    -------
    get_session_from_date(date)
        Get session object from session date
    get_sessions_from_dev_stage_range(dev_stage_range)
        Get sessions from developmental stage range
    """

    def __init__(self, path, animal_id, dob) -> None:
        self.path = path
        self.animal_id = animal_id
        self.dob = dob
        self.sessions = self._load_sessions()

    def _get_session_path(self, session_info):
        # Build session path string by joining `self.path` with
        # `session_info['name']`
        return os.path.join(self.path, session_info["name"])

    def _load_sessions(self):
        """Load all experiment sessions

        Returns
        -------
        list of `ngmd.dataset.Session`
            `Session` objects
        """
        # Get paths to all session directories
        session_paths = glob(os.path.join(self.path, "20??*"))

        sessions = []
        for dir_path in session_paths:
            kwargs = {
                "path": dir_path,
                "animal_id": self.animal_id,
                "dob": self.dob,
            }
            session = Session(**kwargs)
            sessions.append(session)

        return sessions

    def get_session_from_date(self, date):
        """Get session object from session date

        Parameters
        ----------
        date : str
            Date of the experiment session, in the format 'YYYYMMDD'.

        Returns
        -------
        `ngmd.dataset.Session`
            Session object, or None if no session is found
        """
        out_session = None
        for session in self.sessions:
            if session.date == date:
                out_session = session
                break
        return out_session

    def get_sessions_from_dev_stage_range(self, dev_stage_range: tuple):
        """Get sessions from developmental stage range

        Parameters
        ----------
        dev_stage_range : tuple
            Tuple of two integers indicating the lower and upper bounds of the
            developmental stage range

        Returns
        -------
        list of `ngmd.dataset.Session`
            `Session` objects. The list is empty if no sessions are found.
        """
        out_sessions = []
        for session in self.sessions:
            if session.dev_stage is not None:
                if (
                    session.dev_stage >= dev_stage_range[0]
                    and session.dev_stage <= dev_stage_range[1]
                ):
                    out_sessions.append(session)
        return out_sessions


class Session:
    """Experiment session class

    Parameters
    ----------
    path : str
        Path to the directory `'<animal_id>/<date>/'` containing
        the session files
    animal_id : str
        Identification number of the animal used in the experiment session.
    dob : str
        Date of birth of the animal, in the format 'YYYYMMDD'

    Attributes
    ----------
    path : str
        Path to the directory `'<animal_id>/<date>/'` containing
        the session files
    data_path : str
        Path to the subdirectory ending in `'suite2p/plane0'` where useful data
        is located
    animal_id : str
        Identification number of the animal used in the experiment session.
    date : str
        Date of the experiment session, in the format 'YYYYMMDD'.
    dev_stage : str
        Developmental stage of the animal in this session, in units of PDays.
    sample_rate : float
        Sampling rate of the calcium fluorescence traces in this session (Hz)

    Methods
    -------
    load_array(name)
        Load npy array from file
    load_cell_mask()
        Load mask classifying cells and non-cells
    load_fluorescence(remove_non_cells=True, baseline_corrected=False,
                      method='mixture_model')
        Wrapper to load the fluorescence traces array
    load_corrs(subset='all_states', return_p_vals=False, rebuild=False)
        Wrapper to load the precomputed neuronal trace correlations
    load_zscores(subset='all_states', rebuild=False)
        Wrapper to load precomputed correlation z-scores
    """

    def __init__(self, path, animal_id, dob) -> None:
        # Base session path
        self.path = path

        # Record animal ID, session date, and the animal's developmental stage
        self.animal_id = animal_id
        self.date = self._get_date()
        self.dev_stage = self._get_dev_stage(dob)

        # 2p calcium imaging frame rate (Hz)
        self.sample_rate = 30  # TODO: get this from metadata in the future

    def _get_date(self):
        """Get date from the YAML metadata file within the session directory"""
        # Get the path to the YAML file
        yaml_path = glob(os.path.join(self.path, "*.yaml"))[0]

        # Load the YAML file
        with open(yaml_path, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        # Get the date from the 'date' key
        date = yaml_data["date"]

        return date

    def _get_dev_stage(self, dob):
        """Get developmental stage (PDays) from session date and animal's date
        of birth
        """
        # Convert date strings to datetime objects
        date = datetime.strptime(self.date, "%Y%m%d")
        dob = datetime.strptime(dob, "%Y%m%d")

        # Compute the difference in days between the session date and the
        # animal's date of birth
        dev_stage = (date - dob).days

        return dev_stage

    def load_corr_zscore_file(self):
        """Load the npy file containing the correlation/zscore matrices

        Returns
        -------
        array_like
            An N-by-N correlation array, where N is the number of neuronal
            cells identified in the pre-processing suite.
        array_like
            An N-by-N array with corresponding p-values.
        array_like
            An N-by-N array of z-scores, where N is the number of neuronal
            cells identified in the pre-processing suite.
        """
        corr_mat, pval_mat, zscore_mat = np.load(
            os.path.join(self.path, "allcell_clean_corr_pval_zscore.npy")
        )

        # Find rows in corr_mat that have only NaN values. These correspond to
        # ROIs that were seemed not useful in the preprocessing pipeline
        nan_rows = np.where(np.isnan(corr_mat).all(axis=1))[0]

        # Remove these rows from the corr_mat, pval_mat, and zscore_mat
        corr_mat = np.delete(corr_mat, nan_rows, axis=0)
        corr_mat = np.delete(corr_mat, nan_rows, axis=1)
        pval_mat = np.delete(pval_mat, nan_rows, axis=0)
        pval_mat = np.delete(pval_mat, nan_rows, axis=1)
        zscore_mat = np.delete(zscore_mat, nan_rows, axis=0)
        zscore_mat = np.delete(zscore_mat, nan_rows, axis=1)

        return corr_mat, pval_mat, zscore_mat

    def load_corrs(self):
        """Wrapper to load the precomputed neuronal trace correlations

        Returns
        -------
        array_like
            An N-by-N correlation array, where N is the number of neuronal
            cells identified in the pre-processing suite. If the correlation
            matrix does not exist, returns None and raises a warning.

        Notes
        -----
        The correlations were pre-computed as follows. Each neuron's calcium
        fluorescence trace was baseline-corrected and binarized, yielding a
        binary time-series of "calcium events". Then, these time-series are
        binned to 1 second windows and one assigns to each window the count of
        calcium events therein. Finally, for each pair of neurons, one computes
        the correlation coefficient between their binned calcium event count
        time-series.
        """
        try:
            # Try to load the precomputed correlation matrix from disk
            corrs, _, _ = self.load_corr_zscore_file()

        except FileNotFoundError:
            # If the correlation matrix does not exist, return None
            # and raise a warning
            corrs = None
            warnings.warn(
                "Correlation matrix for session {} does not exist".format(
                    self.animal_id + "/" + self.date
                )
            )
        return corrs

    def load_zscores(self):
        """Wrapper to load precomputed correlation z-scores

        Returns
        -------
        array_like
            An N-by-N array of z-scores, where N is the number of neuronal
            cells identified in the pre-processing suite. If the z-score matrix
            does not exist, returns None and raises a warning.

        Notes
        -----
        The z-scores were pre-computed as follows. For each pair of neurons,
        one first computes a correlation coefficient just as in
        `ngmd.dataset.Session.load_corrs`. Then, one produces several random
        circular shifts of the binned calcium event time-series of one of the
        neurons and computes a new correlation coefficient between the shifted
        and unchanged time-series. The z-score is then computed as the number
        of standard deviations away from the mean of the original correlation
        coefficient in the distribution of correlation coefficients obtained
        from the random circular shifts.

        One may interpret a high z-score as a sign of a functional connection
        between the two corresponding neurons, as it means that these two
        neurons are more correlated than what would be expected by chance among
        all possible circular shifts of the neurons' calcium fluorescence
        time-series.
        """
        try:
            # Try to load the precomputed correlation matrix from disk
            _, _, zscores = self.load_corr_zscore_file()

        except FileNotFoundError:
            # If the zscore matrix does not exist, return None
            # and raise a warning
            zscores = None
            warnings.warn(
                "Z-score matrix for session {} does not exist".format(
                    self.animal_id + "/" + self.date
                )
            )
        return zscores

    def load_binarized_traces(self, clean=True):
        """Load session's pre-computed binarized calcium fluorescence traces

        Parameters
        ----------
        clean : bool, optional
            Whether to clean up the loaded binarized traces, by default True. If True, the following operations are performed:
                - remove gel drying cells,
                - remove silent ROIs,
                - remove ROIs marked as outliers

        Returns
        -------
        array_like
            An N-by-T array of binarized calcium fluorescence traces, where N
            is the number of ROIs and T is the number of time points in the recording. If the
            binarized traces do not exist, returns None
        """
        # Get the path to the binarized traces file
        file_path = os.path.join(self.path, "F_upphase.npy")

        # Try to load the binarized traces from disk
        try:
            binarized_traces = np.load(file_path, allow_pickle=True)
        except FileNotFoundError:
            warnings.warn(
                "Binarized traces for session {} do not exist".format(
                    self.animal_id + "/" + self.date
                )
            )
            return None

        # Remove gel drying cells and silent ROIs, if requested
        if clean:
            binarized_traces = self._remove_gel_drying_cells(binarized_traces)
            binarized_traces = binarized_traces[np.sum(binarized_traces, axis=1) > 0]

        return binarized_traces

    def load_gel_drying_mask(self):
        """Load session's pre-computed gel drying mask

        Returns
        -------
        array_like
            An N-by-T array of cell drying masks, where N
            is the number of neuronal cells identified in the pre-processing
            suite and T is the number of time points in the recording. If the
            cell drying mask does not exist, returns None
        """
        # Get the path to the cell drying file
        file_path = os.path.join(self.path, "cell_drying.npy")

        # Try to load the cell drying array from disk
        try:
            cell_drying = np.load(file_path, allow_pickle=True)
        except FileNotFoundError:
            return None

        return cell_drying

    def _remove_gel_drying_cells(self, cell_array):
        """Remove gel drying cells from the binarized traces"""
        # Load the cell drying mask
        cell_drying = self.load_gel_drying_mask()

        # Remove gel drying cells
        cleaned_cell_array = cell_array[~cell_drying, :]

        return cleaned_cell_array

    def load_sttc_array_file(self):
        """Load the npz file containing the session's STTC matrix and
        ancillary percentiles matrix

        Returns
        -------
        array_like
            An N-by-N STTC array, where N is the number of neurons
            recorded in the session.
        array_like
            An N-by-N array with corresponding fence values.
        float
            The synchronicity window (in seconds) used for STTC
            computation.
        int
            The number of trials used to compute the percentiles matrix.
        int
            The seed used to generate the random surrogate STTC matrices.
        """
        arrays = np.load(os.path.join(self.path, "sttc_arrays.npz"))

        sttc_mat = arrays["sttc_mat"]
        percentiles_mat = arrays["percentiles_mat"]
        dt = float(arrays["dt"])
        num_trials = int(arrays["num_trials"])
        seed = int(arrays["seed"])

        return sttc_mat, percentiles_mat, dt, num_trials, seed

    def load_sttc_array(self):
        """Wrapper to load the session's precomputed STTC matrix

        Returns
        -------
        array_like
            An N-by-N STTC array, where N is the number of neurons
            recorded in the session.
        """
        sttc_mat, _, _, _, _ = self.load_sttc_array_file()
        return sttc_mat

    def load_sttc_percentiles(self):
        """Wrapper to load the session's precomputed STTC percentiles matrix

        Returns
        -------
        array_like
            An N-by-N array with corresponding percentiles values.

        Notes
        -----
        Each value in the percentiles matrix represents how far the corresponding STTC value is from the third quartile of the surrogate STTC values distribution, in
        inter-quartile-range units. In symbols,
        `percentiles_mat[i, j] = (sttc_mat[i, j] - Q3) / IQR`. This makes it easy for one to
        find pairs of sttc values that are larger than one would expect (outlier) and
        could, thus, imply a connection between the respective ROIs. For example, the
        mask `percentiles_mat > 1.5` points at all pairs of ROIs whose STTC values lie
        beyond the 'inner fence' in Tukey's percentiles method. Similarly, `percentiles_mat > 3`
        points at all 'outer fence' outliers
        """
        _, percentiles_mat, _, _, _ = self.load_sttc_array_file()
        return percentiles_mat
