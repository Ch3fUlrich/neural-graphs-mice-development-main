import matplotlib.pyplot as plt
import numpy as np

from ngmd import permutation_tests


def test_difference_of_means_supports_arrays_and_axis():
    first = np.array([[1, 2], [3, 4]])
    second = np.array([[2, 3], [4, 5]])

    np.testing.assert_array_equal(
        permutation_tests.difference_of_means(first, second, axis=0),
        np.array([-1.0, -1.0]),
    )


def test_pairwise_hypothesis_testing_returns_expected_result_shape():
    measurements = {"a": np.array([1, 1, 1]), "b": np.array([3, 3, 3])}

    results = permutation_tests.pairwise_hypothesis_testing(
        measurements,
        n_resamples=9,
        alpha=0.05,
        method="fdr_bh",
    )

    assert results["label_pairs"] == [("a", "b")]
    assert len(results["test_stats"]) == 1
    assert len(results["p_vals"]) == 1
    assert len(results["reject_null"]) == 1


def test_plot_helpers_return_axes():
    measurements = {"a": np.array([1, 1, 1]), "b": np.array([3, 3, 3])}
    ax, results = permutation_tests.plot_stat_diff(
        measurements,
        return_all=True,
        n_resamples=9,
        alpha=0.05,
        method="fdr_bh",
    )

    assert ax.get_figure() is not None
    assert results["label_pairs"] == [("a", "b")]

    class Result:
        null_distribution = np.array([0, 1, 2])
        statistic = 1

    fig, ax = plt.subplots()
    returned = permutation_tests.plot_permutation_distribution(Result(), ax=ax)

    assert returned is ax
