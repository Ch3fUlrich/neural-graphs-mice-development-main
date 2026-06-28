import numpy as np

from ngmd import sttc


def test_convolve2d_applies_kernel_with_padding():
    image = np.array([[1, 2], [3, 4]])
    kernel = np.array([[1, 0], [0, 1]])

    result = sttc.convolve2d(image, kernel, padding=1)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [1, 2, 0],
                [3, 5, 2],
                [0, 3, 4],
            ],
            dtype=np.float32,
        ),
    )


def test_tile_spike_trains_expands_spikes_by_window():
    spikes = np.array([[0, 1, 0, 0]])

    tiled = sttc.tile_spike_trains(spikes, dt_in_samples=1)

    np.testing.assert_array_equal(tiled, np.array([[True, True, True, False]]))


def test_proportions_and_time_coverage_for_simple_spike_trains():
    spikes = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    tiled = sttc.tile_spike_trains(spikes, dt_in_samples=1)

    proportions = sttc.proportions_of_matched_spikes(spikes, tiled)
    coverage = sttc.fraction_of_time_covered(tiled)

    np.testing.assert_allclose(proportions, np.ones((2, 2)))
    np.testing.assert_allclose(coverage, np.array([0.75, 0.75]))


def test_matrix_spike_time_tiling_coefficient_matches_known_values():
    spike_matrix = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    sttc_mat = sttc.matrix_spike_time_tiling_coefficient(spike_matrix, 1, 1)

    np.testing.assert_allclose(
        sttc_mat,
        np.array(
            [
                [1.0, 0.7, -0.33333333, np.nan],
                [0.7, 1.0, 0.54347826, np.nan],
                [-0.33333333, 0.54347826, 1.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        ),
        equal_nan=True,
    )
