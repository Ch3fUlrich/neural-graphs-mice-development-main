"""Plotting module"""

import time

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import ticker

from ngmd.graphs import remove_small_components


class UnibasColors:
    """Official color palette of the University of Basel

    Attributes
    ----------
    mint : str
        Mint color
    mint_hell : str
        Light mint color
    rot : str
        Red color
    anthrazit : str
        Anthrazit color
    anthrazit_hell : str
        Light anthrazit color
    """

    def __init__(self) -> None:
        self.mint = "#a5D7D2"
        self.mint_hell = "#d2ebe9"
        self.rot = "#d20537"
        self.anthrazit = "#2d373c"
        self.anthrazit_hell = "#46505a"


def plot_nx_graph(
    G_nx,
    pos=None,
    gamma=1.0,
    ax=None,
):
    """Plot a networkx graph

    Parameters
    ----------
    G_nx : `networkx` graph
        The graph to plot
    pos : dict, optional
        A dictionary of node positions, by default None. If None, a spring
        layout is computed
    gamma : float, optional
        Gamma correction for edge thicknesses, by default 1.
    ax : `matplotlib.axes.Axes`, optional
        The axes to plot on, by default None. If None, a new figure is created

    Returns
    -------
    `matplotlib.axes.Axes`
        The axes on which the graph was plotted
    """
    # Parse node positions
    if pos is None:
        pos = nx.spring_layout(G_nx)

    edge_thicknesses = 1
    edgelist = G_nx.edges()

    # Plot nodes then edges
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 7))
    nx.draw_networkx_nodes(
        G_nx,
        pos,
        nodelist=G_nx.nodes(),
        node_size=15,
        node_color="black",
        alpha=0.7,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G_nx,
        pos,
        edgelist=edgelist,
        width=edge_thicknesses,
        edge_color="red",
        alpha=0.5,
        ax=ax,
    )

    return ax


def plot_binary_traces(binary_traces, sample_rate=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Plot binary traces as a binary image of shape (N, T)
    ax.imshow(binary_traces, cmap="binary", aspect="auto", interpolation="none")

    # If sample rate is provided, convert x-tick labels to time stamps in the
    # format MM:SS, otherwise plot a time-step axis
    if sample_rate is not None:

        def format_func(x, pos):
            return time.strftime("%M:%S", time.gmtime(x // sample_rate))

        formatter = ticker.FuncFormatter(format_func)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel("Time (s)")
    else:
        ax.set_xlabel("Time step")

    # Set y-axis name
    ax.set_ylabel("Neuron ID")

    # Y-axis ticks should be integers, because they represent neuron IDs
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    return ax


def networks_side_by_side(networks_dict, min_cc_size=3, figsize=(18, 6)):
    """Plot networks side by side

    Parameters
    ----------
    networks_dict : list of dict
        List of dictionaries, each containing the keys
        - 'graph': a `ngmd.graphs.SimilarityGraph` object
        - 'animal_id': animal ID
        - 'session_date': session date
    min_cc_size : int, optional
        Minimum size of connected components to display, by default 3
    figsize : tuple, optional
        Figure size, by default (18, 6)

    Returns
    -------
    tuple
        A `matplotlib.figure.Figure` and a `matplotlib.axes.Axes` object for
        the plot
    """
    fig, axs = plt.subplots(1, len(networks_dict), figsize=figsize)
    for i, network in enumerate(networks_dict):
        G = remove_small_components(network["graph"].G_nx, min_cc_size)
        axs[i].set_title("{}/{} ".format(network["animal_id"], network["session_date"]))
        plot_nx_graph(G, ax=axs[i])
    return fig, axs
