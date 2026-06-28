"""
Data generation module.
Contains functions to extract various statistics and metrics from the dataset,
refactored from the standalone scripts.
"""

import logging
import os
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from ngmd import (
    config,
    dataset,
    graphs,
    structural_consistency as sc,
    sttc,
    timeseries,
    utils,
)

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- network_summaries utilities ---

NETWORK_CONFIG = config.get_config_section(
    "network_summaries",
    {
        "num_subsample_trials": 20,
        "subsample_size": 250,
    },
)
NUM_SUBSAMPLE_TRIALS = NETWORK_CONFIG["num_subsample_trials"]
SUBSAMPLE_SIZE = NETWORK_CONFIG["subsample_size"]


def mat_is_valid(sim_mat, subsample_size):
    if sim_mat is None:
        return False
    elif subsample_size is not None and len(sim_mat) < subsample_size:
        return False
    else:
        return True


def subsample_and_compute(G_nx, fun, subsample_size, num_subsample_trials):
    values = []
    if subsample_size is not None:
        for _ in range(num_subsample_trials):
            G_nx_sub = graphs.subsample_graph(G_nx, subsample_size)
            values.append(fun(G_nx_sub))
    else:
        values.append(fun(G_nx))
    return values


def _initialize_results_df():
    return pd.DataFrame(
        columns=[
            "animal_id",
            "session_date",
            "dev_stage",
            "num_neurons",
            "num_conn_comp",
            "density",
            "transitivity",
            "avg_clust",
            "size_largest_cc",
            "avg_short_path_length",
            "radius",
            "diameter",
            "median_page_rank",
        ]
    )


def _compute_network_metrics(G_nx):
    metrics = {}
    metrics["num_conn_comp"] = nx.number_connected_components(G_nx)
    metrics["density"] = nx.density(G_nx)
    metrics["transitivity"] = nx.transitivity(G_nx)
    metrics["avg_clust"] = nx.algorithms.approximation.average_clustering(G_nx)
    largest_cc = G_nx.subgraph(max(nx.connected_components(G_nx), key=len)).copy()
    metrics["size_largest_cc"] = len(largest_cc)
    metrics["avg_short_path_length"] = nx.average_shortest_path_length(largest_cc)
    metrics["radius"] = nx.radius(largest_cc)
    metrics["diameter"] = nx.diameter(largest_cc)
    metrics["median_page_rank"] = np.median(list(nx.pagerank(G_nx).values()))
    return metrics


def _extract_typical_values(list_of_metrics):
    keys = list_of_metrics[0].keys()
    typical_values = {}
    for key in keys:
        values = [metrics[key] for metrics in list_of_metrics]
        typical_values[key] = np.median(values)
    return typical_values


def _loop_computation(i, sessions_to_loop, sim_type, connect_thresh):
    session = sessions_to_loop[i]
    sim_mat = utils.select_sim_mat(session, sim_type)

    if mat_is_valid(sim_mat, SUBSAMPLE_SIZE):
        sg = graphs.SimilarityGraph(sim_mat, connect_thresh)
        network_metrics = subsample_and_compute(
            sg.G_nx,
            _compute_network_metrics,
            SUBSAMPLE_SIZE,
            NUM_SUBSAMPLE_TRIALS,
        )
        typical_values = _extract_typical_values(network_metrics)
        loop_results = [
            session.animal_id,
            session.date,
            session.dev_stage,
            len(sim_mat),
        ]
        loop_results.extend([typical_values[key] for key in typical_values.keys()])
        return loop_results
    else:
        return None


def _save_to_file(results_df, save_dir, sim_type, connect_thresh):
    file_name = "network_summaries_{}".format(sim_type)
    file_name += "_connect_thresh={:.1f}".format(connect_thresh)
    if SUBSAMPLE_SIZE is not None:
        file_name += "_subsample_size={}".format(SUBSAMPLE_SIZE)
    file_name += ".csv"
    file_path = os.path.join(save_dir, file_name)
    results_df.to_csv(file_path, index=False)


# --- MAIN FUNCTIONS ---


def generate_sim_mat_statistics(data_dir, save_dir, num_bins=100, sim_type="sttc"):
    """Compiles statistics of the similarity matrices in the dataset"""
    def compute_histogram(values, num_bins):
        counts, bins = np.histogram(values, bins=num_bins, range=(-1.0, 1.0))
        counts = counts.astype(float).tolist()
        bins = bins.astype(float).tolist()
        return counts, bins

    animal_dataset = dataset.AnimalDataset(data_dir)
    results = pd.DataFrame(
        columns=[
            "animal_id",
            "session_date",
            "dev_stage",
            "num_neurons",
            "sim_counts",
            "sim_bins",
            "min_sim_val",
            "max_sim_val",
            "avg_sim_val",
            "std_sim_val",
        ]
    )

    for animal in animal_dataset.animals:
        logging.info("Processing animal {}".format(animal.animal_id))
        for session in tqdm(animal.sessions):
            sim_mat = utils.select_sim_mat(session, sim_type)
            if sim_mat is not None:
                non_trivial_sim_vals, num_neurons = utils.get_non_trivial_values(sim_mat)
                counts, bins = compute_histogram(non_trivial_sim_vals, num_bins)
                new_row = [
                    session.animal_id,
                    session.date,
                    session.dev_stage,
                    num_neurons,
                    counts,
                    bins,
                    np.min(non_trivial_sim_vals),
                    np.max(non_trivial_sim_vals),
                    np.mean(non_trivial_sim_vals).astype("float"),
                    np.std(non_trivial_sim_vals).astype("float"),
                ]
                results.loc[len(results)] = new_row

    results.to_csv(os.path.join(save_dir, "{}_statistics.csv".format(sim_type)), index=False)


def generate_degree_distributions(data_dir, save_dir, connect_thresh, sim_type="sttc_percentiles"):
    """Compiles list of graph degrees for each session, given a connectivity threshold"""
    animal_dataset = dataset.AnimalDataset(data_dir)
    results = pd.DataFrame(
        columns=[
            "animal_id",
            "session_date",
            "dev_stage",
            "degrees",
        ]
    )

    for animal in animal_dataset.animals:
        logging.info("Processing animal {}".format(animal.animal_id))
        for session in tqdm(animal.sessions):
            sim_mat = utils.select_sim_mat(session, sim_type)
            if sim_mat is not None:
                sg = graphs.SimilarityGraph(similarity_matrix=sim_mat, connect_thresh=connect_thresh)
                degrees = [float(d) for (_, d) in sg.G_nx.degree()]
                new_row = [
                    session.animal_id,
                    session.date,
                    session.dev_stage,
                    degrees,
                ]
                results.loc[len(results)] = new_row

    file_path = os.path.join(
        save_dir,
        "degree_distributions_{}_".format(sim_type)
        + "connect_thresh={:.1f}.csv".format(connect_thresh),
    )
    results.to_csv(file_path, index=False)


def generate_laplacian_spectra(data_dir, save_dir, connect_thresh, sim_type="sttc_percentiles"):
    """Compiles list of Laplacian eigenvalues (spectrum) for the graph of each session"""
    animal_dataset = dataset.AnimalDataset(data_dir)
    results = pd.DataFrame(
        columns=[
            "animal_id",
            "session_date",
            "dev_stage",
            "eigenvalues",
        ]
    )

    for animal in animal_dataset.animals:
        logging.info("Processing animal {}".format(animal.animal_id))
        for session in tqdm(animal.sessions):
            sim_mat = utils.select_sim_mat(session, sim_type)
            if sim_mat is not None:
                sg = graphs.SimilarityGraph(similarity_matrix=sim_mat, connect_thresh=connect_thresh)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    eigenvalues = nx.normalized_laplacian_spectrum(sg.G_nx)
                eigenvalues = eigenvalues.astype(float).tolist()
                new_row = [
                    session.animal_id,
                    session.date,
                    session.dev_stage,
                    eigenvalues,
                ]
                results.loc[len(results)] = new_row

    file_path = os.path.join(
        save_dir,
        "laplacian_spectra_{}_".format(sim_type)
        + "connect_thresh={:.1f}.csv".format(connect_thresh),
    )
    results.to_csv(file_path, index=False)


def generate_network_summaries(data_dir, save_dir, connect_thresh, sim_type="sttc_percentiles"):
    """Compiles summaries of networks created from the correlation matrices in the dataset"""
    animal_dataset = dataset.AnimalDataset(data_dir)
    results_df = _initialize_results_df()
    sessions_to_loop = utils.get_sessions_to_loop(animal_dataset)

    results_list = Parallel(n_jobs=-1, verbose=50)(
        delayed(_loop_computation)(i, sessions_to_loop, sim_type, connect_thresh)
        for i in range(len(sessions_to_loop))
    )

    for row in results_list:
        if row is not None:
            results_df.loc[len(results_df)] = row

    _save_to_file(results_df, save_dir, sim_type, connect_thresh)
    logging.info("Done! Results saved to {}".format(save_dir))


STRUCTURAL_CONFIG = config.get_config_section(
    "structural_consistency",
    {
        "num_trials": 20,
        "num_subsample_trials": 20,
        "subsample_size": 250,
    },
)
_SC_NUM_TRIALS = STRUCTURAL_CONFIG["num_trials"]
_SC_NUM_SUBSAMPLE_TRIALS = STRUCTURAL_CONFIG["num_subsample_trials"]
_SC_SUBSAMPLE_SIZE = STRUCTURAL_CONFIG["subsample_size"]


def _compile_sc_statistics(trial_scs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.mean(trial_scs)
        std = np.std(trial_scs)
        upper = np.percentile(trial_scs, 95)
        lower = np.percentile(trial_scs, 5)
    return mean, std, upper, lower


def generate_structural_consistency(data_dir, save_dir, sim_type, connect_thresh):
    """Computes structural consistency for every session and saves results to CSV.

    Parameters
    ----------
    data_dir : str
        Path to the dataset directory.
    save_dir : str
        Directory where the output CSV file will be written.
    sim_type : str
        Similarity-matrix type to use (e.g. ``"sttc_percentiles"``, ``"zscores"``).
    connect_thresh : float
        Connectivity threshold for creating graphs from similarity matrices.
    """
    animal_dataset = dataset.AnimalDataset(data_dir)

    results = pd.DataFrame(
        columns=[
            "animal_id",
            "session_date",
            "dev_stage",
            "mean_sc",
            "std_sc",
            "upper_sc",
            "lower_sc",
        ]
    )

    for animal in animal_dataset.animals:
        logging.info("Processing animal {}".format(animal.animal_id))
        for session in tqdm(animal.sessions):
            sim_mat = utils.select_sim_mat(session, sim_type)

            if mat_is_valid(sim_mat, _SC_SUBSAMPLE_SIZE):
                sg = graphs.SimilarityGraph(
                    similarity_matrix=sim_mat, connect_thresh=connect_thresh
                )

                def trial_fun():
                    return subsample_and_compute(
                        sg.G_nx,
                        sc.structural_consistency,
                        _SC_SUBSAMPLE_SIZE,
                        _SC_NUM_SUBSAMPLE_TRIALS,
                    )

                trial_scs = Parallel(n_jobs=-1, verbose=1)(
                    delayed(trial_fun)() for _ in range(_SC_NUM_TRIALS)
                )
                trial_scs = [v for sublist in trial_scs for v in sublist]

                mean, std, upper, lower = _compile_sc_statistics(trial_scs)
                results.loc[len(results)] = [
                    session.animal_id,
                    session.date,
                    session.dev_stage,
                    mean,
                    std,
                    upper,
                    lower,
                ]

    file_name = "structural_consistency_{}".format(sim_type)
    file_name += "_connect_thresh={:.1f}".format(connect_thresh)
    if _SC_SUBSAMPLE_SIZE is not None:
        file_name += "_subsample_size={}".format(_SC_SUBSAMPLE_SIZE)
    file_name += ".csv"
    file_path = os.path.join(save_dir, file_name)
    results.to_csv(file_path, index=False)
    logging.info("Done! Results saved to {}".format(save_dir))


def generate_binarized_traces_statistics(data_dir, save_dir):
    """Compiles statistics of the binarized traces in the dataset"""
    def compile_pulse_lists(animal_dataset):
        pulse_rates = []
        pulse_widths = []
        pulse_intervals = []

        for animal in animal_dataset.animals:
            logging.info("Processing animal {}".format(animal.animal_id))

            for session in animal.sessions:
                binarized_traces = session.load_binarized_traces()
                n_rois, _ = binarized_traces.shape

                for roi in range(n_rois):
                    info = timeseries.gather_binary_trace_statistics(
                        binarized_traces[roi], session.sample_rate
                    )
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
                                "dev_stage": [session.dev_stage] * len(info["pulse_widths"]),
                            }
                        )
                    )
                    pulse_intervals.append(
                        pd.DataFrame(
                            {
                                "animal_id": [animal.animal_id] * len(info["pulse_intervals"]),
                                "date": [session.date] * len(info["pulse_intervals"]),
                                "roi": [roi] * len(info["pulse_intervals"]),
                                "pulse_intervals": info["pulse_intervals"],
                                "dev_stage": [session.dev_stage] * len(info["pulse_intervals"]),
                            }
                        )
                    )
        return pulse_rates, pulse_widths, pulse_intervals

    animal_dataset = dataset.AnimalDataset(data_dir)
    pulse_rates, pulse_widths, pulse_intervals = compile_pulse_lists(animal_dataset)

    pulse_rates = pd.concat(pulse_rates, ignore_index=True)
    pulse_widths = pd.concat(pulse_widths, ignore_index=True)
    pulse_intervals = pd.concat(pulse_intervals, ignore_index=True)

    pulse_rates["roi"] = pulse_rates["roi"].astype(int)
    pulse_widths["roi"] = pulse_widths["roi"].astype(int)
    pulse_intervals["roi"] = pulse_intervals["roi"].astype(int)

    pulse_rates.to_csv(os.path.join(save_dir, "pulse_rates.gz"), index=False, compression="gzip")
    pulse_widths.to_csv(os.path.join(save_dir, "pulse_widths.gz"), index=False, compression="gzip")
    pulse_intervals.to_csv(os.path.join(save_dir, "pulse_intervals.gz"), index=False, compression="gzip")

    logging.info("Done. Results saved to {}".format(save_dir))


# --- small_worldness ---


def generate_small_worldness(data_dir, save_dir, sim_type, connect_thresh):
    """Computes small-world coefficient (omega) for each session and saves results to CSV.

    Parameters
    ----------
    data_dir : str
        Path to the dataset directory.
    save_dir : str
        Directory where the output CSV file will be written.
    sim_type : str
        Similarity-matrix type (e.g. ``"sttc_percentiles"``, ``"zscores"``).
    connect_thresh : float
        Connectivity threshold for creating graphs from similarity matrices.
    """
    animal_dataset = dataset.AnimalDataset(data_dir)
    sessions_to_loop = utils.get_sessions_to_loop(animal_dataset)

    results_df = pd.DataFrame(
        columns=[
            "animal_id",
            "session_date",
            "dev_stage",
            "small_worldness",
        ]
    )

    # Compute omega serially so that the internal parallelism of omega() is used
    for i, session in enumerate(sessions_to_loop):
        sim_mat = utils.select_sim_mat(session, sim_type)
        if sim_mat is not None:
            sg = graphs.SimilarityGraph(
                similarity_matrix=sim_mat, connect_thresh=connect_thresh
            )
            largest = graphs.largest_cc(sg.G_nx)
            try:
                small_worldness = graphs.omega(largest, nrand=5, n_jobs=-1, verbose=50, timeout=180)
            except nx.exception.NetworkXError:
                small_worldness = np.nan

            results_df.loc[len(results_df)] = [
                session.animal_id,
                session.date,
                session.dev_stage,
                small_worldness,
            ]

    file_path = os.path.join(
        save_dir,
        "small_worldness_{}_".format(sim_type)
        + "connect_thresh={:.1f}.csv".format(connect_thresh),
    )
    results_df.to_csv(file_path, index=False)
    logging.info("Done! Results saved to {}".format(save_dir))


# --- precompute_sttc_arrays ---

PRECOMPUTE_CONFIG = config.get_config_section(
    "precompute_sttc_arrays",
    {
        "seed": 2024,
        "dt": 0.25,
        "method": "pulse_shuffle",
        "num_trials": 3,
        "n_jobs": 1,
        "force_recompute": False,
    },
)
_STTC_SEED = PRECOMPUTE_CONFIG["seed"]
_STTC_DT = PRECOMPUTE_CONFIG["dt"]
_STTC_METHOD = PRECOMPUTE_CONFIG["method"]
_STTC_NUM_TRIALS = PRECOMPUTE_CONFIG["num_trials"]
_STTC_N_JOBS = PRECOMPUTE_CONFIG["n_jobs"]
_STTC_FORCE_RECOMPUTE = PRECOMPUTE_CONFIG["force_recompute"]


def compute_sttc_arrays(animal_dataset, session):
    """Compute and save STTC arrays for a single session.

    The result is written to ``<session.path>/sttc_arrays.npz``.
    Skips sessions that already have the file unless ``force_recompute`` is
    set in the config.
    """
    force_recompute = _STTC_FORCE_RECOMPUTE

    save_file_path = os.path.join(session.path, "sttc_arrays.npz")
    if os.path.exists(save_file_path) and not force_recompute:
        return

    binary_traces = animal_dataset.load_traces_from_good_rois(session)

    sttc_mat = sttc.matrix_spike_time_tiling_coefficient(
        binary_traces, session.sample_rate, _STTC_DT
    )

    surrogate_sttc_values = timeseries.compile_surrogate_sttc_values(
        binary_traces,
        session.sample_rate,
        method=_STTC_METHOD,
        num_trials=_STTC_NUM_TRIALS,
        n_jobs=_STTC_N_JOBS,
        dt=_STTC_DT,
    )

    percentiles_mat = timeseries.compute_percentiles_matrix(
        np.abs(sttc_mat), np.abs(surrogate_sttc_values)
    )

    logging.info("Saving results...")
    np.savez_compressed(
        save_file_path,
        sttc_mat=sttc_mat,
        percentiles_mat=percentiles_mat,
        dt=_STTC_DT,
        num_trials=_STTC_NUM_TRIALS,
        method=_STTC_METHOD,
        seed=_STTC_SEED,
    )


def generate_sttc_arrays(data_dir, n_sessions_in_parallel=1):
    """Precompute STTC arrays for all sessions in the dataset.

    Parameters
    ----------
    data_dir : str
        Path to the dataset directory. STTC arrays are saved directly into
        each session's sub-directory as ``sttc_arrays.npz``.
    n_sessions_in_parallel : int, optional
        Number of sessions to process in parallel (default: 1).
    """
    np.random.seed(_STTC_SEED)
    animal_dataset = dataset.AnimalDataset(data_dir)
    sessions_to_loop = utils.get_sessions_to_loop(animal_dataset)

    Parallel(n_jobs=n_sessions_in_parallel, verbose=50)(
        delayed(compute_sttc_arrays)(animal_dataset, session)
        for session in sessions_to_loop
    )
