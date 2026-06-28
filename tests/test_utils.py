import numpy as np
import pandas as pd
import pytest

from ngmd import config, utils


def test_keep_strings_ending_in_filters_matching_suffixes():
    values = ["alpha.csv", "beta.txt", "gamma.csv", "delta.tsv"]

    assert utils.keep_strings_ending_in(values, [".csv", ".tsv"]) == [
        "alpha.csv",
        "gamma.csv",
        "delta.tsv",
    ]


def test_labels_from_cuts_formats_neighboring_intervals():
    assert utils.labels_from_cuts([0, 15, 24]) == ["PDay 0-15", "PDay 15-24"]


def test_select_sim_mat_dispatches_to_session_loader():
    class Session:
        def load_corrs(self):
            return "corrs"

        def load_zscores(self):
            return "zscores"

        def load_sttc_array(self):
            return "sttc"

        def load_sttc_percentiles(self):
            return "sttc_percentiles"

    session = Session()

    assert utils.select_sim_mat(session, "corrs") == "corrs"
    assert utils.select_sim_mat(session, "zscores") == "zscores"
    assert utils.select_sim_mat(session, "sttc") == "sttc"
    assert utils.select_sim_mat(session, "sttc_percentiles") == "sttc_percentiles"
    with pytest.raises(ValueError, match="Invalid similarity type"):
        utils.select_sim_mat(session, "unknown")


def test_fill_sym_mat_from_triu_builds_symmetric_matrix():
    matrix = utils.fill_sym_mat_from_triu(3, [1, 2, 3], diag=9)

    np.testing.assert_array_equal(
        matrix,
        np.array(
            [
                [9, 1, 2],
                [1, 9, 3],
                [2, 3, 9],
            ]
        ),
    )


def test_update_table_creates_appends_and_deduplicates(tmp_path):
    save_path = tmp_path / "bad_rois.csv"
    first = pd.DataFrame({"animal_id": ["A"], "date": ["20240101"], "roi": [1]})
    second = pd.DataFrame(
        {
            "animal_id": ["A", "B"],
            "date": ["20240101", "20240102"],
            "roi": [1, 3],
        }
    )

    utils.update_table(first, first.columns, save_path)
    utils.update_table(second, second.columns, save_path)

    saved = pd.read_csv(save_path, dtype={"date": str})
    assert saved.to_dict("records") == [
        {"animal_id": "A", "date": "20240101", "roi": 1},
        {"animal_id": "B", "date": "20240102", "roi": 3},
    ]


def test_get_non_trivial_values_removes_diagonal_and_nans():
    matrix = np.array(
        [
            [1.0, 0.2, np.nan],
            [0.2, 1.0, 0.4],
            [np.nan, 0.4, 1.0],
        ]
    )

    values, num_neurons = utils.get_non_trivial_values(matrix)

    assert num_neurons == 3
    np.testing.assert_array_equal(values, np.array([0.2, 0.2, 0.4, 0.4]))


def test_get_sessions_to_loop_flattens_animals():
    class Animal:
        def __init__(self, sessions):
            self.sessions = sessions

    class Dataset:
        animals = [Animal(["a", "b"]), Animal(["c"])]

    assert utils.get_sessions_to_loop(Dataset()) == ["a", "b", "c"]


def test_get_default_dev_stage_cuts_uses_config(monkeypatch):
    monkeypatch.setattr(
        config,
        "get_config_section",
        lambda section, defaults=None: {"cuts": [1, 2, 3]},
    )

    assert utils.get_default_dev_stage_cuts() == [1, 2, 3]
