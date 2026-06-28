import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ngmd import graphs, plotting


def test_unibas_colors_exposes_expected_palette_values():
    palette = plotting.UnibasColors()

    assert palette.rot == "#d20537"
    assert palette.anthrazit == "#2d373c"


def test_plot_nx_graph_and_binary_traces_return_axes():
    fig, axes = plt.subplots(1, 2)
    graph_ax = plotting.plot_nx_graph(nx.path_graph(3), ax=axes[0])
    traces_ax = plotting.plot_binary_traces(np.array([[0, 1], [1, 0]]), ax=axes[1])

    assert graph_ax is axes[0]
    assert traces_ax is axes[1]
    assert traces_ax.get_xlabel() == "Time step"


def test_plot_binary_traces_formats_time_axis_when_sample_rate_is_given():
    ax = plotting.plot_binary_traces(np.array([[0, 1, 0, 1]]), sample_rate=2)

    assert ax.get_xlabel() == "Time (s)"


def test_networks_side_by_side_plots_each_network():
    networks = [
        {
            "graph": graphs.SimilarityGraph(np.array([[0.0, 1.0], [1.0, 0.0]]), 0.5),
            "animal_id": "DON-1",
            "session_date": "20240101",
        },
        {
            "graph": graphs.SimilarityGraph(np.array([[0.0, 1.0], [1.0, 0.0]]), 0.5),
            "animal_id": "DON-2",
            "session_date": "20240102",
        },
    ]

    fig, axes = plotting.networks_side_by_side(networks, min_cc_size=1)

    assert fig is not None
    assert len(axes) == 2
    assert axes[0].get_title() == "DON-1/20240101 "
