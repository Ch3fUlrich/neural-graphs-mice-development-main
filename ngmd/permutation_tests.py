"""Permutation tests"""

import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests

# ~~~ Test statistics ~~~ #


def difference_of_means(sample_1, sample_2, axis=0):
    """Difference of means test statistic

    Parameters
    ----------
    sample_1 : array_like
        Numbers in the first sample
    sample_2 : array_like
        Numbers in the second sample
    axis : int, optional
        Direction of the arrays onto which to gather the means, by default 0

    Returns
    -------
    float
        Difference of means in the two samples
    """
    return np.mean(sample_1, axis=axis) - np.mean(sample_2, axis=axis)


# ~~~ Permutation tests ~~~ #


def pairwise_hypothesis_testing(
    measurement_dict,
    test_statistic=None,
    n_resamples=9999,
    **kwargs,
):
    """Run pairwise hypothesis testing

    Parameters
    ----------
    measurement_dict : dict
        Dictionary of measurements. Each key should represent one sample in the
        pairwise testing
    test_statistic : callable, optional
        Test statistic to compare a pair of samples, by default
        `nb_utils.difference_of_means`
    n_resamples : int, optional
        Number of resamples when doing a permutation test for each pair of
        samples, by default 9999
    **kwargs : dict, optional
        Optional keyword arguments for the `statsmodels.stats.multitest.multipletests` method, for instance `alpha` and `method`.

    Returns
    -------
    dict
        Results dictionary, containing the label pairs that were tested, their
        corresponding test statistics, their (multiple-test-corrected) p-values
        and a list of which null hypothesis can be rejected

    See also
    --------
    `statsmodels.stats.multitest.multipletests`
    """
    if test_statistic is None:
        test_statistic = difference_of_means
    labels = list(measurement_dict.keys())

    # Pairwise testing
    label_pairs = list(combinations(labels, 2))
    p_vals = []
    test_stats = []
    for pair in label_pairs:
        res = permutation_test(
            (measurement_dict[pair[0]], measurement_dict[pair[1]]),
            test_statistic,
            vectorized=True,
            n_resamples=n_resamples,
            alternative="two-sided",
        )
        p_vals.append(res.pvalue)
        test_stats.append(res.statistic)

    # Correct for multiple testing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reject, p_vals_corrected, _, _ = multipletests(p_vals, **kwargs)

    # Assemble results into dictionary
    results = {
        "label_pairs": label_pairs,
        "test_stats": test_stats,
        "p_vals": p_vals_corrected,
        "reject_null": reject,
    }

    return results


# ~~~ Plotting functionality ~~~ #


def plot_permutation_distribution(res, ax=None, bins=51, fontsize=12):
    """Plot permutation distribution

    Parameters
    ----------
    res : `PermutationTestResult`
        Permutation test result, as returned by `scipy.stats.permutation_test`
    ax : `matplotlib.Axis`, optional
        Axis onto which to make the plot. If None, a new figure/axis will be
        created, by default None
    bins : int, optional
        Number of bins in the permutation distribution histogram, by default 51
    fontsize : int, optional
        Font size, by default 12

    Returns
    -------
    `matplotlib.Axis`
        Axis onto which the plot was made
    """
    # Parse axis
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Plot histogram os test statistics across different permutations
    ax.hist(res.null_distribution, bins=bins, color="#46505a")

    # Overlay test statistic of the original permutation as a vertical line
    ax.axvline(x=res.statistic, color="#d20537", label="Original statistic")

    # Set titles and labels
    ax.set_title("Permutation distribution of test statistic", fontsize=fontsize)
    ax.set_xlabel("Value of Statistic", fontsize=fontsize)
    ax.set_ylabel("Frequency", fontsize=fontsize)
    ax.legend(fontsize=fontsize)

    return ax


def plot_stat_diff(measurement_dict, ax=None, return_all=False, **kwargs):
    """Plot statistical differences between measured samples

    Parameters
    ----------
    measurement_dict : dict
        Dictionary of measurements. Each key should represent one sample in the
        pairwise testing
    ax : `matplotlib.Axis`, optional
        Axis onto which to make the plot, by default a new figure is created
    return_all : bool, optional
        Return plot and test results, by default only the plot is returned
    **kwargs: dict, optional
        Optional keyword arguments for `nb_utils.pairwise_hypothesis_testing`

    Returns
    -------
    tuple
        If `return_all is True`, then this tuple contains the `matplotlib.Axis`
        object onto which the plot was made and a dictionary with the results
        of the pairwise testing procedure that led to the plot. See
        `nb_utils.pairwise_hypothesis_testing`
    `matplotlib.Axis`
        If `return_all is False`. This is the object onto which the plot was
        made.
    """
    # Parse axis
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Perform pairwise testing on the labels in the measurement dictionary
    results = pairwise_hypothesis_testing(
        measurement_dict,
        **kwargs,
    )

    # Get labels
    labels = list(measurement_dict.keys())
    n_labels = len(labels)

    # Create dictionaries to convert label names to array indices and
    # vice-versa
    idx2label = dict(enumerate(labels))
    label2idx = dict([(value, key) for key, value in idx2label.items()])

    # Populate array of statistical difference markers
    stat_diff_markers = np.zeros((n_labels, n_labels))
    for idx, pair in enumerate(results["label_pairs"]):
        i, j = (label2idx[pair[0]], label2idx[pair[1]])
        if results["reject_null"][idx]:
            stat_diff_markers[i][j] = results["test_stats"][idx]
    stat_diff_markers -= (stat_diff_markers).T
    mask = stat_diff_markers == 0.0
    stat_diff_markers = np.ma.array(stat_diff_markers, mask=mask)

    # Plot the statistical differences array as an image
    cmap = "PiYG"
    im = ax.imshow(stat_diff_markers, cmap=cmap, interpolation="none")

    # Set ticks
    ticks_loc = list(np.arange(n_labels))
    ax.set_xticks(ticks_loc)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(ticks_loc)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)

    _ = plt.colorbar(
        im,
        ax=ax,
        label="Significant differences (row vs. column)",
    )

    if return_all:
        return ax, results
    else:
        return ax
