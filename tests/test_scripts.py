"""Tests for data-generation functions in ngmd.data_generation.

All generate_* functions and their private helpers now live exclusively in
ngmd.data_generation; the scripts/data_generation/ files are thin CLI wrappers
with no logic of their own.  Tests therefore import from ngmd.data_generation
and monkeypatch its namespace directly.
"""

import os

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import ngmd.data_generation as dg


# ---------------------------------------------------------------------------
# mat_is_valid / subsample_and_compute (shared helpers)
# ---------------------------------------------------------------------------

def test_mat_is_valid():
    assert dg.mat_is_valid(None, 2) is False
    assert dg.mat_is_valid(np.ones((2, 2)), 3) is False
    assert dg.mat_is_valid(np.ones((2, 2)), None) is True
    assert dg.mat_is_valid(np.ones((4, 4)), 3) is True


def test_subsample_and_compute(monkeypatch):
    graph = nx.path_graph(4)
    monkeypatch.setattr(dg.graphs, "subsample_graph", lambda G, size: G.subgraph([0, 1]))

    values = dg.subsample_and_compute(
        graph,
        lambda G: G.number_of_nodes(),
        subsample_size=2,
        num_subsample_trials=3,
    )
    assert values == [2, 2, 2]

    # subsample_size=None → run once on the full graph
    values_full = dg.subsample_and_compute(
        graph,
        lambda G: G.number_of_nodes(),
        subsample_size=None,
        num_subsample_trials=99,
    )
    assert values_full == [4]


# ---------------------------------------------------------------------------
# generate_sim_mat_statistics: histogram helper (now an inner function)
# ---------------------------------------------------------------------------

def test_sim_mat_statistics_histogram_uses_fixed_range(monkeypatch, tmp_path):
    """Verify the histogram inside generate_sim_mat_statistics uses range(-1,1)."""

    class Session:
        animal_id = "DON-1"
        date = "20240101"
        dev_stage = 12

    class Animal:
        animal_id = "DON-1"
        sessions = [Session()]

    class MockDataset:
        animals = [Animal()]

    sim_mat = np.array([[0.0, -0.5, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
    monkeypatch.setattr(dg.dataset, "AnimalDataset", lambda path: MockDataset())
    monkeypatch.setattr(dg.utils, "select_sim_mat", lambda session, sim_type: sim_mat)
    monkeypatch.setattr(dg.utils, "get_non_trivial_values", lambda m: (np.array([-0.5, 0.0, 0.5]), 3))

    dg.generate_sim_mat_statistics(str(tmp_path), str(tmp_path), num_bins=4, sim_type="sttc")

    out_path = tmp_path / "sttc_statistics.csv"
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert len(df) == 1
    import ast
    counts = ast.literal_eval(df.iloc[0]["sim_counts"])
    bins = ast.literal_eval(df.iloc[0]["sim_bins"])
    assert counts == [0.0, 1.0, 1.0, 1.0]
    assert bins == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# generate_network_summaries helpers
# ---------------------------------------------------------------------------

def test_network_summary_initialize_results_df():
    df = dg._initialize_results_df()
    assert list(df.columns) == [
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


def test_network_summary_compute_metrics():
    graph = nx.complete_graph(4)
    metrics = dg._compute_network_metrics(graph)
    assert metrics["num_conn_comp"] == 1
    assert metrics["density"] == pytest.approx(1.0)
    assert metrics["size_largest_cc"] == 4
    assert metrics["radius"] == 1
    assert metrics["diameter"] == 1


def test_network_summary_extract_typical_values():
    typical = dg._extract_typical_values(
        [{"a": 1, "b": 10}, {"a": 3, "b": 20}, {"a": 100, "b": 30}]
    )
    assert typical == {"a": pytest.approx(3.0), "b": pytest.approx(20.0)}


def test_network_summary_save_path(tmp_path, monkeypatch):
    monkeypatch.setattr(dg, "SUBSAMPLE_SIZE", 7)
    dg._save_to_file(
        pd.DataFrame({"animal_id": ["A"]}),
        tmp_path,
        sim_type="zscores",
        connect_thresh=4,
    )
    assert (tmp_path / "network_summaries_zscores_connect_thresh=4.0_subsample_size=7.csv").exists()


def test_network_summary_loop_computation(monkeypatch):
    class Session:
        animal_id = "DON-1"
        date = "20240101"
        dev_stage = 12

    sim_mat = np.array([[0.0, 1.0], [1.0, 0.0]])
    monkeypatch.setattr(dg, "SUBSAMPLE_SIZE", None)
    monkeypatch.setattr(dg.utils, "select_sim_mat", lambda session, sim_type: sim_mat)
    monkeypatch.setattr(
        dg,
        "subsample_and_compute",
        lambda G, fun, subsample_size, num_trials: [
            {"metric_a": 1, "metric_b": 5},
            {"metric_a": 3, "metric_b": 7},
        ],
    )

    result = dg._loop_computation(0, [Session()], "zscores", 0.5)
    assert result[:3] == ["DON-1", "20240101", 12]
    assert result[3] == 2  # len(sim_mat)


# ---------------------------------------------------------------------------
# generate_structural_consistency helpers
# ---------------------------------------------------------------------------

def test_compile_sc_statistics_with_percentiles():
    mean, std, upper, lower = dg._compile_sc_statistics([1, 2, 3])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(np.std([1, 2, 3]))
    assert upper == pytest.approx(2.9)
    assert lower == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# compute_sttc_arrays (migrated from precompute_sttc_arrays.py)
# ---------------------------------------------------------------------------

def test_compute_sttc_arrays_writes_npz(monkeypatch, tmp_path):
    monkeypatch.setattr(dg, "_STTC_FORCE_RECOMPUTE", True)
    monkeypatch.setattr(dg, "_STTC_DT", 0.25)
    monkeypatch.setattr(dg, "_STTC_METHOD", "pulse_shuffle")
    monkeypatch.setattr(dg, "_STTC_NUM_TRIALS", 3)
    monkeypatch.setattr(dg, "_STTC_N_JOBS", 1)
    monkeypatch.setattr(dg, "_STTC_SEED", 2024)

    class MockAnimalDataset:
        def load_traces_from_good_rois(self, session):
            return np.array([[0, 1, 0], [1, 0, 1]])

    class MockSession:
        path = str(tmp_path)
        sample_rate = 30

    sttc_mat = np.array([[1.0, 0.2], [0.2, 1.0]])
    percentiles = np.array([[np.nan, 95.0], [95.0, np.nan]])

    monkeypatch.setattr(
        dg.sttc,
        "matrix_spike_time_tiling_coefficient",
        lambda traces, sample_rate, dt: sttc_mat,
    )
    monkeypatch.setattr(
        dg.timeseries,
        "compile_surrogate_sttc_values",
        lambda traces, sample_rate, method, num_trials, n_jobs, dt: np.array([[0.1]]),
    )
    monkeypatch.setattr(
        dg.timeseries,
        "compute_percentiles_matrix",
        lambda original, surrogate: percentiles,
    )

    dg.compute_sttc_arrays(MockAnimalDataset(), MockSession())

    saved = np.load(tmp_path / "sttc_arrays.npz")
    np.testing.assert_array_equal(saved["sttc_mat"], sttc_mat)
    np.testing.assert_array_equal(saved["percentiles_mat"], percentiles)
    assert float(saved["dt"]) == pytest.approx(0.25)
    assert int(saved["num_trials"]) == 3
    assert str(saved["method"]) == "pulse_shuffle"
    assert int(saved["seed"]) == 2024


def test_compute_sttc_arrays_skips_if_file_exists(monkeypatch, tmp_path):
    """Verify that an existing .npz file is skipped when force_recompute is False."""
    monkeypatch.setattr(dg, "_STTC_FORCE_RECOMPUTE", False)

    existing = tmp_path / "sttc_arrays.npz"
    existing.touch()

    called = []
    monkeypatch.setattr(
        dg.sttc,
        "matrix_spike_time_tiling_coefficient",
        lambda *a, **kw: called.append(1) or np.zeros((2, 2)),
    )

    class MockSession:
        path = str(tmp_path)
        sample_rate = 30

    dg.compute_sttc_arrays(object(), MockSession())
    assert called == [], "Should have skipped computation"


# ---------------------------------------------------------------------------
# generate_binarized_traces_statistics: compile_pulse_lists (now inner)
# ---------------------------------------------------------------------------

def test_binarized_traces_statistics_output_shape(monkeypatch, tmp_path):
    """End-to-end: verify the three .gz files are written with correct metadata."""

    class Session:
        animal_id = "DON-1"
        date = "20240101"
        dev_stage = 12
        sample_rate = 2

        def load_binarized_traces(self):
            return np.array([[0, 1, 1, 0], [0, 0, 1, 0]])

    class Animal:
        animal_id = "DON-1"
        sessions = [Session()]

    class MockDataset:
        animals = [Animal()]

    monkeypatch.setattr(dg.dataset, "AnimalDataset", lambda path: MockDataset())

    dg.generate_binarized_traces_statistics(str(tmp_path), str(tmp_path))

    rates_df = pd.read_csv(tmp_path / "pulse_rates.gz", compression="gzip")
    widths_df = pd.read_csv(tmp_path / "pulse_widths.gz", compression="gzip")

    assert rates_df["animal_id"].tolist() == ["DON-1", "DON-1"]
    assert widths_df["dev_stage"].tolist() == [12, 12]


# ---------------------------------------------------------------------------
# generate_small_worldness helpers
# ---------------------------------------------------------------------------

def test_generate_small_worldness_saves_csv(monkeypatch, tmp_path):
    class Session:
        animal_id = "DON-1"
        date = "20240101"
        dev_stage = 12

    class MockDataset:
        animals = []  # not used directly

    sim_mat = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    monkeypatch.setattr(dg.dataset, "AnimalDataset", lambda path: MockDataset())
    monkeypatch.setattr(dg.utils, "get_sessions_to_loop", lambda ds: [Session()])
    monkeypatch.setattr(dg.utils, "select_sim_mat", lambda session, sim_type: sim_mat)
    monkeypatch.setattr(dg.graphs, "omega", lambda G, **kwargs: 0.25)

    dg.generate_small_worldness(str(tmp_path), str(tmp_path), "zscores", 0.5)

    out_path = tmp_path / "small_worldness_zscores_connect_thresh=0.5.csv"
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert df["small_worldness"].iloc[0] == pytest.approx(0.25)


def test_generate_small_worldness_nan_on_networkx_error(monkeypatch, tmp_path):
    class Session:
        animal_id = "DON-1"
        date = "20240101"
        dev_stage = 12

    class MockDataset:
        animals = []

    sim_mat = np.array([[0.0, 1.0], [1.0, 0.0]])
    monkeypatch.setattr(dg.dataset, "AnimalDataset", lambda path: MockDataset())
    monkeypatch.setattr(dg.utils, "get_sessions_to_loop", lambda ds: [Session()])
    monkeypatch.setattr(dg.utils, "select_sim_mat", lambda session, sim_type: sim_mat)
    monkeypatch.setattr(
        dg.graphs, "omega",
        lambda G, **kwargs: (_ for _ in ()).throw(nx.exception.NetworkXError("too small"))
    )

    dg.generate_small_worldness(str(tmp_path), str(tmp_path), "zscores", 0.5)

    df = pd.read_csv(tmp_path / "small_worldness_zscores_connect_thresh=0.5.csv")
    assert np.isnan(df["small_worldness"].iloc[0])
