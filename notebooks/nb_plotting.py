import warnings

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ngmd import plotting, utils


def _draw_vline_with_tag(value, ax, ha="right", **kwargs):
    color = kwargs.pop("color", plotting.UnibasColors().rot)
    ax.axvline(value, color=color, **kwargs)

    offset = -0.03 if ha == "right" else 0.03
    ax.text(
        offset + value,
        0.9 * ax.get_ylim()[1],
        f"{value:.2f}",
        color=color,
        ha=ha,
        va="top",
        bbox=dict(boxstyle="square,pad=1", fc=(1, 1, 1, 0.5), ec=(0.0, 0.0, 0.0, 0.5)),
    )


def draw_fence_line(array, ax=None, k=3):
    if ax is None:
        _, ax = plt.subplots()

    array = np.array(array).flatten()
    q1 = np.nanquantile(array, 0.25)
    q3 = np.nanquantile(array, 0.75)
    iqr = q3 - q1
    threshold = q3 + k * iqr

    ax = _draw_vline_with_tag(
        threshold, ax, color=plotting.UnibasColors().rot, linestyle="--", ha="right"
    )
    return ax


def draw_max_line(array, ax=None, ha="right"):
    if ax is None:
        _, ax = plt.subplots()

    array = np.array(array).flatten()
    max_val = np.nanmax(array)

    ax = _draw_vline_with_tag(
        max_val, ax, color=plotting.UnibasColors().anthrazit, linestyle="--", ha="left"
    )
    return ax


def draw_percentile_line(array, pctile, ax=None, ha="right"):
    if ax is None:
        _, ax = plt.subplots()

    array = np.array(array).flatten()
    pctile_val = np.nanpercentile(array, pctile)

    ax = _draw_vline_with_tag(
        pctile_val,
        ax,
        color=plotting.UnibasColors().anthrazit,
        linestyle="--",
        ha=ha,
    )
    return ax


def draw_median_line(values, ax):
    palette = plotting.UnibasColors()
    median = np.median(values)
    ax.axvline(
        median, color=palette.rot, alpha=0.6, label="Median = {:.2f}".format(median)
    )
    ax.legend()


def plot_histogram(values, ax=None, **kwargs):
    """Plot a histogram of values

    Parameters
    ----------
    values : array-like
        Values to plot
    ax : `matplotlib.axes.Axes`, optional
        The axes to plot on, by default None. If None, a new figure is created
    **kwargs : dict, optional
        Optional keyword arguments for `matplotlib.axes.Axes.hist`


    Returns
    -------
    `matplotlib.axes.Axes`
        The axis onto which the plot was made
    `numpy.ndarray`
        The densities of the histogram
    `numpy.ndarray`
        The bins of the histogram

    See Also
    --------
    matplotlib.axes.Axes.hist
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Define some default values in kwargs
    color = kwargs.pop("color", plotting.UnibasColors().anthrazit_hell)
    bins = kwargs.pop("bins", 99)
    density = kwargs.pop("density", True)
    histtype = kwargs.pop("histtype", "bar")

    # Plot the histogram
    densities, bins, _ = ax.hist(
        values,
        bins=bins,
        density=density,
        color=color,
        histtype=histtype,
        **kwargs,
    )

    return ax, densities, bins


def _plot_freq_histogram(bins, freqs, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Parse kwargs
    color = kwargs.pop("color", "black")
    alpha = kwargs.pop("alpha", 0.6)
    kwargs.pop("bar", None)
    ax.plot(
        bins,
        freqs,
        color=color,
        alpha=alpha,
        **kwargs,
    )

    ax.set(ylabel="Frequency")

    return ax


def plot_sim_histogram(df, ax=None, **kwargs):
    """Plot histogram of similarity values

    Parameters
    ----------
    df : `pandas.DataFrame`
        The input data frame. It should contain columns named `'sim_counts'`,
        `'sim_bins'` and `'num_neurons'`. See the output of
        `scrips/dataset_statistics.py`
    ax : `matplotlib.Axes`, optional
        The axis onto which to make the plot, by default a new figure/axis will
        be created
    **kwargs : dict, optional
        Any extra keyword arguments that can be passed to `matplotlib.Axes.plot`,
        for example `color` or `label`

    Returns
    -------
    `matplotlib.Axes`
        The axis onto which the plot was made
    """

    counts = np.array(df["sim_counts"].to_list()).sum(axis=0)
    num_sim_values = (df["num_neurons"] * (df["num_neurons"] - 1)).sum()
    global_freqs = counts / num_sim_values
    bins = df["sim_bins"].iloc[0][:-1]

    ax = _plot_freq_histogram(
        bins,
        global_freqs,
        ax=ax,
        **kwargs,
    )

    return ax


def sessions_box_plot(animal, sim_type="corrs", showfliers=True):
    """Make boxplots of neuron similarity values for each experiment session in
    which the specified animal participated

    Parameters
    ----------
    animal : `ngmd.dataset.Animal`
        The input animal
    sim_type : str, optional
        Type of similarity values to plot, by default 'corrs'. Can be either
        'corrs' or 'zscores'
    showfliers : bool, optional
        Show fliers on the boxplots, by default True

    Returns
    -------
    `matplotlib.Axes`
        The axis onto which the plot was made
    """
    columns = []
    pdays = []
    for session in animal.sessions:
        sim_mat = utils.select_sim_mat(session, sim_type)

        if sim_mat is not None:
            non_trivial_sim_vals, _ = utils.get_non_trivial_values(sim_mat)
            columns.append(non_trivial_sim_vals)
            pdays.append(session.dev_stage)

    _, ax = plt.subplots(figsize=(12, 4))
    ax.boxplot(columns, showfliers=showfliers)
    plt.xticks(np.arange(len(columns)) + 1, pdays)
    ax.set_xlabel("PDay")
    return ax


def sessions_num_neurons_plot(animal, figsize=(12, 4)):
    """Make bar plots of the number of neurons identified in each experiment
    session in which the specified animal participated

    Parameters
    ----------
    animal : `ngmd.dataset.Animal`
        The input animal
    figsize : tuple, optional
        Figure dimensions, by default (12, 4)

    Returns
    -------
    `matplotlib.Axes`
        The axis onto which the plot was made
    """
    num_neurons = []
    pdays = []
    for session in animal.sessions:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_mat = session.load_corrs()
        if corr_mat is not None:
            num_neurons.append(len(corr_mat))
            pdays.append(session.dev_stage)

    _, ax = plt.subplots(figsize=figsize)
    ax.bar(pdays, num_neurons)
    plt.xticks(pdays, pdays)
    ax.set_xlabel("PDay")
    ax.set_ylabel("Number of neurons identified in the session")
    ax.set_title("{}".format(animal.animal_id))

    return ax


def boxplot_summary(summary_dict, bootstrap=5000, showfliers=True):
    """Make boxplots of network summaries

    Parameters
    ----------
    summary_dict : dict
        Dictionary of summaries. Should have connectivity thresholds as keys
        and summary numbers as values
    bootstrap : int, optional
        How many bootstrap samples to draw in order to define the limits of the
        boxplots, by default 5000
    showfliers : bool, optional
        Show fliers on the boxplots, by default True

    Returns
    -------
    `matplotlib.Axes`
        The axis onto which the plot was made
    """
    _, axes = plt.subplots(1, len(summary_dict), figsize=(28, 4), sharey=True)
    for i, key in enumerate(summary_dict):
        try:
            ax = axes[i]
        except TypeError:
            ax = axes

        # Get non-NaN values
        values = list(summary_dict[key].values())
        values = [v[~np.isnan(v)] for v in values]

        ax.boxplot(values, bootstrap=bootstrap, showfliers=showfliers, sym="+")
        ax.set(axisbelow=True, title="connect_thresh={}".format(key))

        ax.set_xticklabels(summary_dict[key].keys(), rotation=45)

        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

    return axes


def plot_degree_histogram(df, **kwargs):
    """Plot histogram of network degrees

    Parameters
    ----------
    df : `pandas.DataFrame`
        The input data frame. Must have columns `'degrees'`, and `'degrees_w'`
    **kwargs : dict, optional
        Optional keyword arguments for `ngmd.plotting.plot_histogram`

    Returns
    -------
    `matplotlib.Axes`
        The axis onto which the plot was made
    """
    degrees = df["degrees"].tolist()
    values = np.array([item for sublist in degrees for item in sublist])
    ax, _, _ = plot_histogram(values, **kwargs)
    return ax


def plot_spectrum_curves(df, ax=None, ignore_zero_eigvals=False, **kwargs):
    """Make plot of Laplacian eigenvalue curves in the specified data frame

    Parameters
    ----------
    df : `pandas.DataFrame`
        The input data frame. Must contain a column `'eigenvalues'`
    ax : `matplotlib.Axis`, optional
        The axis on which to make the plot, by default a new figure/axis will
        be created
    ignore_zero_eigvals : bool, optional
        Discard the leading zero eigenvalues from the plot, by default False
    **kwargs : dict, optional
        Optional keyword arguments for `matplotlib.Axes.plot`

    Returns
    -------
    `matplotlib.Axes`
        The axis onto which the plot was made
    """
    # Parse default kwargs
    color = kwargs.pop("color", plotting.UnibasColors().anthrazit_hell)
    alpha = kwargs.pop("alpha", 0.7)
    label = kwargs.pop("label", None)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    spectra = df["eigenvalues"].tolist()
    n_spectra = len(spectra)
    for i, spectrum in enumerate(spectra):
        spectrum = np.array(spectrum)

        if ignore_zero_eigvals:
            spectrum = spectrum[spectrum > 1e-3]
            spectrum = np.concatenate(([0], spectrum))

        x_axis = np.linspace(0, 1, len(spectrum))

        if i != (n_spectra - 1):
            ax.plot(x_axis, spectrum, color=color, alpha=alpha, **kwargs)
        else:
            ax.plot(x_axis, spectrum, color=color, alpha=alpha, label=label, **kwargs)

    return ax


def _add_error_band_traces(fig, data):
    # Get x-values
    x = list(data["x"])

    # Set upper and lower y-values
    y_upper = list(data["y"] + data["error_y"]["array"])
    if data["error_y"]["arrayminus"] is None:
        y_lower = list(data["y"] - data["error_y"]["array"])
    else:
        y_lower = list(data["y"] - data["error_y"]["arrayminus"])

    # Set color of the error band as a transparent version of the line
    # color
    color = tuple(
        int(data["line"]["color"].lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)
    )
    color = f"rgba({color},.3)"
    color = color.replace("((", "(")
    color = color.replace("),", ",")
    color = color.replace(" ", "")

    # Add the error band traces to the figure
    # We need to include the traces in reverse order as well so that
    # they are plotted correctly, as explained here:
    # https://plotly.com/python/continuous-error-bars/
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=y_upper + y_lower[::-1],  # upper, then lower reversed
            fill="toself",
            fillcolor=color,
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=data["legendgroup"],
            xaxis=data["xaxis"],
            yaxis=data["yaxis"],
        )
    )
    return fig


def _reorder_data(fig):
    reordered_data = []
    for i in range(int(len(fig.data) / 2)):
        reordered_data.append(fig.data[i + int(len(fig.data) / 2)])
        reordered_data.append(fig.data[i])
    fig.data = tuple(reordered_data)
    return fig


def _parse_error_y_mode(error_y_mode):
    ERROR_MODES = {"bar", "band", "bars", "bands", None}

    if error_y_mode not in ERROR_MODES:
        raise ValueError(
            "'error_y_mode' must be one of {}".format(ERROR_MODES)
            + ", received {}.".format(error_y_mode)
        )


def line_with_interval(error_y_mode=None, **kwargs):
    """Extension of `plotly.express.line` to use interval (error) bands

    Parameters
    ----------
    interval_y_mode : str, optional
        One of {'bar','band','bars','bands',None}, by default None. If 'bar' or
        'bars', the error bars are plotted as vertical lines. If 'band' or
        'bands', the error bars are plotted as filled areas.
    **kwargs : dict
        Keyword arguments to pass to `plotly.express.line`

    Returns
    -------
    `plotly.graph_objects.Figure`
        The figure containing the line plot with error bars

    Raises
    ------
    ValueError
    If `interval_y_mode` is not one of {'bar','band','bars','bands',None}

    See Also
    --------
    plotly.express.line

    Notes
    -----
    This function is a modified version of the code found in this
    StackOverflow answer:
    https://stackoverflow.com/a/69594497/9474857
    """
    _parse_error_y_mode(error_y_mode)

    if error_y_mode in {"bar", "bars", None}:
        fig = px.line(**kwargs)  # Use the default plotly.express.line function

    elif error_y_mode in {"band", "bands"}:
        if "error_y" not in kwargs:
            raise ValueError("You must also provide an 'error_y' argument.")

        # Initialize a temporary figure as a regular line plot with error bars
        figure_with_error_bars = px.line(**kwargs)

        # Initialize the final figure as a copy of the temporary figure without
        # the 'error_y' argument
        fig = px.line(**{arg: val for arg, val in kwargs.items() if arg != "error_y"})

        # Add the error bands as filled areas
        for data in figure_with_error_bars.data:
            fig = _add_error_band_traces(fig, data)

        # Reorder data as explained in this StackOverflow answer:
        # https://stackoverflow.com/a/66854398/8849755
        fig = _reorder_data(fig)

    return fig
