import json

import numpy as np
import pandas as pd
import pytest
import yaml

from ngmd import dataset


def _write_yaml(path, content):
    path.write_text(yaml.safe_dump(content), encoding="utf-8")


@pytest.fixture
def session_dir(tmp_path):
    path = tmp_path / "DON-000001" / "20240110"
    path.mkdir(parents=True)
    _write_yaml(path / "session.yaml", {"date": "20240110"})
    return path


def test_get_animal_paths_finds_don_directories_under_years(tmp_path):
    (tmp_path / "2024" / "DON-000001").mkdir(parents=True)
    (tmp_path / "misc" / "DON-ignored").mkdir(parents=True)

    assert dataset.get_animal_paths(str(tmp_path)) == [
        str(tmp_path / "2024" / "DON-000001")
    ]


def test_load_dev_stages_reads_json_or_returns_empty_dict(tmp_path):
    json_path = tmp_path / "dev_stages.json"
    json_path.write_text(json.dumps({"DON-1": {"20240101": 10}}), encoding="utf-8")

    assert dataset.load_dev_stages(json_path) == {"DON-1": {"20240101": 10}}
    assert dataset.load_dev_stages(tmp_path / "missing.json") == {}


def test_session_loaders_read_and_clean_small_files(session_dir):
    session = dataset.Session(str(session_dir), animal_id="DON-000001", dob="20240101")

    corr = np.array(
        [
            [1.0, 0.5, np.nan],
            [0.5, 1.0, np.nan],
            [np.nan, np.nan, np.nan],
        ]
    )
    pvals = np.array(
        [
            [0.0, 0.1, np.nan],
            [0.1, 0.0, np.nan],
            [np.nan, np.nan, np.nan],
        ]
    )
    zscores = np.array(
        [
            [0.0, 3.0, np.nan],
            [3.0, 0.0, np.nan],
            [np.nan, np.nan, np.nan],
        ]
    )
    np.save(session_dir / "allcell_clean_corr_pval_zscore.npy", np.array([corr, pvals, zscores]))

    loaded_corrs, loaded_pvals, loaded_zscores = session.load_corr_zscore_file()
    np.testing.assert_array_equal(loaded_corrs, corr[:2, :2])
    np.testing.assert_array_equal(loaded_pvals, pvals[:2, :2])
    np.testing.assert_array_equal(loaded_zscores, zscores[:2, :2])
    np.testing.assert_array_equal(session.load_corrs(), corr[:2, :2])
    np.testing.assert_array_equal(session.load_zscores(), zscores[:2, :2])

    traces = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    np.save(session_dir / "F_upphase.npy", traces)
    np.save(session_dir / "cell_drying.npy", np.array([False, True, False]))

    np.testing.assert_array_equal(session.load_binarized_traces(clean=False), traces)
    np.testing.assert_array_equal(
        session.load_binarized_traces(clean=True),
        np.array([[1, 0, 0]]),
    )

    sttc_mat = np.array([[1.0, 0.2], [0.2, 1.0]])
    percentiles_mat = np.array([[np.nan, 95.0], [95.0, np.nan]])
    np.savez_compressed(
        session_dir / "sttc_arrays.npz",
        sttc_mat=sttc_mat,
        percentiles_mat=percentiles_mat,
        dt=0.25,
        num_trials=3,
        seed=2024,
    )

    loaded = session.load_sttc_array_file()
    np.testing.assert_array_equal(loaded[0], sttc_mat)
    np.testing.assert_array_equal(loaded[1], percentiles_mat)
    assert loaded[2:] == (0.25, 3, 2024)
    np.testing.assert_array_equal(session.load_sttc_array(), sttc_mat)
    np.testing.assert_array_equal(session.load_sttc_percentiles(), percentiles_mat)


def test_session_missing_matrix_loaders_warn_and_return_none(session_dir):
    session = dataset.Session(str(session_dir), animal_id="DON-000001", dob="20240101")

    with pytest.warns(UserWarning, match="Correlation matrix"):
        assert session.load_corrs() is None
    with pytest.warns(UserWarning, match="Z-score matrix"):
        assert session.load_zscores() is None


def test_animal_and_dataset_load_minimal_structure(tmp_path):
    animal_dir = tmp_path / "DON-000001"
    session_path = animal_dir / "20240110"
    session_path.mkdir(parents=True)
    _write_yaml(
        animal_dir / "DON-000001.yaml",
        {
            "cohort_year": "2024",
            "animal_id": "DON-000001",
            "dob": "20240101",
            "implanted": True,
            "injected": True,
            "sex": "F",
        },
    )
    _write_yaml(session_path / "session.yaml", {"date": "20240110"})

    animal_dataset = dataset.AnimalDataset(str(tmp_path))

    animal = animal_dataset.get_animal_from_id("DON-000001")
    assert animal is not None
    assert animal_dataset.get_animal_from_id("missing") is None
    assert animal.get_session_from_date("20240110").dev_stage == 9
    assert animal.get_session_from_date("missing") is None
    assert animal.get_sessions_from_dev_stage_range((0, 10))[0].date == "20240110"
    assert animal_dataset.get_sessions_from_dev_stage_range((0, 10))[0].date == "20240110"


def test_bad_measurement_tables_and_good_roi_loading(tmp_path):
    animal_dataset = object.__new__(dataset.AnimalDataset)
    animal_dataset.path = str(tmp_path)
    animal_dataset.bad_rois = pd.DataFrame(
        {"animal_id": ["DON-000001"], "date": ["20240110"], "roi": [1]}
    )

    with pytest.raises(ValueError, match="animal_id"):
        animal_dataset.update_bad_measurements(pd.DataFrame({"date": ["20240110"]}))
    with pytest.raises(ValueError, match="date"):
        animal_dataset.update_bad_measurements(pd.DataFrame({"animal_id": ["DON-000001"]}))

    bad_session = pd.DataFrame({"animal_id": ["DON-000001"], "date": ["20240110"]})
    animal_dataset.update_bad_measurements(bad_session)
    assert (tmp_path / "bad_sessions.csv").exists()
    assert animal_dataset.load_bad_measurements_table("sessions").shape == (1, 2)
    with pytest.raises(ValueError, match="which"):
        animal_dataset.load_bad_measurements_table("unknown")

    class Session:
        animal_id = "DON-000001"
        date = "20240110"

        def load_binarized_traces(self, clean=True):
            assert clean is True
            return np.array([[1, 0], [0, 1], [1, 1]])

    np.testing.assert_array_equal(
        animal_dataset.load_traces_from_good_rois(Session()),
        np.array([[1, 0], [1, 1]]),
    )
