import numpy as np
import pytest

from ngmd import timeseries


def test_compute_correlations_matches_numpy_corrcoef():
    traces = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    np.testing.assert_allclose(timeseries.compute_correlations(traces), np.corrcoef(traces))


def test_time_shuffle_with_zero_max_shift_preserves_traces():
    traces = np.array([[0, 1, 0], [1, 0, 1]])

    np.testing.assert_array_equal(
        timeseries.make_time_shuffle_surrogates(traces, max_shift=1),
        traces,
    )


def test_pulse_shuffle_preserves_shape_and_column_activity(monkeypatch):
    traces = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
    monkeypatch.setattr(timeseries.np.random, "permutation", lambda n: np.arange(n)[::-1])

    shuffled = timeseries.make_pulse_shuffle_surrogates(traces)

    assert shuffled.shape == traces.shape
    np.testing.assert_array_equal(shuffled.sum(axis=0), traces.sum(axis=0))


def test_poisson_surrogate_returns_binary_traces_with_same_shape(monkeypatch):
    traces = np.array([[1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]])
    monkeypatch.setattr(timeseries.np.random, "poisson", lambda lam, size: np.ones(size))

    surrogate = timeseries.make_poisson_surrogate_traces(traces)

    assert surrogate.shape == traces.shape
    assert set(np.unique(surrogate)).issubset({0, 1})


def test_make_surrogate_traces_dispatches_and_rejects_unknown_method():
    traces = np.array([[0, 1, 0], [1, 0, 1]])

    np.testing.assert_array_equal(
        timeseries.make_surrogate_traces(traces, method="time_shuffle", max_shift=1),
        traces,
    )
    with pytest.raises(ValueError, match="Invalid method"):
        timeseries.make_surrogate_traces(traces, method="not-a-method")


def test_compile_surrogate_sttc_values_collects_upper_triangular_values(monkeypatch):
    traces = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]])
    sttc_matrix = np.array([[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]])
    monkeypatch.setattr(timeseries, "make_surrogate_traces", lambda traces, method: traces)
    monkeypatch.setattr(
        timeseries.sttc,
        "matrix_spike_time_tiling_coefficient",
        lambda surrogate_traces, sample_rate, dt: sttc_matrix,
    )

    values = timeseries.compile_surrogate_sttc_values(
        traces,
        sample_rate=1,
        method="pulse_shuffle",
        num_trials=2,
        n_jobs=1,
        dt=1,
    )

    np.testing.assert_array_equal(values, np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]))


def test_compute_percentiles_matrix_is_symmetric_with_nan_diagonal():
    original = np.array([[0.0, 0.1, 0.2], [0.1, 0.0, 0.3], [0.2, 0.3, 0.0]])
    surrogate = np.array([[0.0, 0.15, 0.25], [0.4, 0.5, 0.6]])

    percentiles = timeseries.compute_percentiles_matrix(original, surrogate)

    assert np.isnan(np.diag(percentiles)).all()
    np.testing.assert_allclose(percentiles, percentiles.T, equal_nan=True)
    np.testing.assert_allclose(
        percentiles[np.triu_indices(3, 1)],
        np.array([100 * 2 / 9, 100 * 4 / 9, 100 * 6 / 9]),
    )


@pytest.mark.parametrize(
    ("trace", "expected_positive", "expected_negative"),
    [
        ([0, 1, 1, 0, 1], [0, 3], [2, 5]),
        ([1, 1, 0, 0], [0], [1]),
        ([0, 0, 1, 1], [1], [4]),
    ],
)
def test_parse_transitions_handles_edge_pulses(trace, expected_positive, expected_negative):
    positive, negative = timeseries.parse_transitions(np.array(trace))

    np.testing.assert_array_equal(positive, expected_positive)
    np.testing.assert_array_equal(negative, expected_negative)


def test_gather_binary_trace_statistics():
    trace = np.array([0, 1, 1, 0, 1])

    stats = timeseries.gather_binary_trace_statistics(trace, sample_rate=2)

    assert stats["duration"] == 2.5
    assert stats["n_pulses"] == 2
    assert stats["pulse_widths"] == [1.0, 1.0]
    assert stats["pulse_intervals"] == [0.5]
    assert stats["active_fraction"] == 0.8
