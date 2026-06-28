import networkx as nx
import numpy as np
import pytest

from ngmd import graphs


def test_adjacency_from_similarity_thresholds_diagonal_and_binarizes(similarity_matrix):
    weighted = graphs.adjacency_from_similarity(similarity_matrix, connect_thresh=0.5)
    binary = graphs.adjacency_from_similarity(
        similarity_matrix,
        connect_thresh=0.5,
        binarize=True,
    )

    np.testing.assert_array_equal(
        weighted,
        np.array(
            [
                [0.0, 0.8, 0.0],
                [0.8, 0.0, 0.6],
                [0.0, 0.6, 0.0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        binary,
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )


def test_adjacency_from_similarity_rejects_non_square_matrices():
    with pytest.raises(ValueError, match="square"):
        graphs.adjacency_from_similarity(np.ones((2, 3)))


def test_similarity_graph_formats_values_and_recomputes_when_threshold_changes():
    matrix = np.array([[np.nan, -0.8], [np.inf, 0.0]])

    graph = graphs.SimilarityGraph(matrix, connect_thresh=0.5)

    assert graph.similarity_matrix[0, 0] == 0.0
    assert graph.similarity_matrix[0, 1] == 0.8
    assert np.isfinite(graph.similarity_matrix[1, 0])
    assert graph.G_nx.number_of_edges() == 1

    graph.connect_thresh = np.finfo(float).max
    assert graph.G_nx.number_of_edges() == 0


def test_similarity_graph_uses_mean_plus_two_std_threshold_when_none():
    matrix = np.array([[0.0, 0.2, 0.4], [0.2, 0.0, 0.6], [0.4, 0.6, 0.0]])

    graph = graphs.SimilarityGraph(matrix)
    off_diag = matrix[~np.eye(3, dtype=bool)]
    expected = np.mean(off_diag) + 2 * np.std(off_diag)

    assert graph.connect_thresh == pytest.approx(expected)


def test_neural_activity_graph_accepts_corr_matrix_or_activity_traces():
    corr_mat = np.array([[1.0, 0.7], [0.7, 1.0]])
    from_corr = graphs.NeuralActivityGraph(corr_mat=corr_mat, connect_thresh=0.5)

    assert from_corr.G_nx.number_of_edges() == 1

    traces = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    from_traces = graphs.NeuralActivityGraph(activity_traces=traces, connect_thresh=0.5)

    np.testing.assert_allclose(from_traces.corr_mat, np.array([[1.0, -1.0], [-1.0, 1.0]]))
    with pytest.raises(ValueError, match="activity_traces"):
        graphs.NeuralActivityGraph()


def test_component_helpers_and_subsampling(monkeypatch):
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (3, 4)])
    graph.add_node(5)

    assert set(graphs.largest_cc(graph).nodes) == {0, 1, 2}
    assert set(graphs.remove_small_components(graph, min_cc_size=2).nodes) == {
        0,
        1,
        2,
        3,
        4,
    }

    monkeypatch.setattr(graphs.np.random, "choice", lambda nodes, size, replace: [0, 1])
    subgraph = graphs.subsample_graph(graph, 2)

    assert set(subgraph.nodes) == {0, 1}
    assert subgraph.has_edge(0, 1)


def test_omega_uses_reference_graphs(monkeypatch):
    graph = nx.complete_graph(4)
    monkeypatch.setattr(graphs, "random_reference", lambda G, niter: G.copy())
    monkeypatch.setattr(graphs, "lattice_reference", lambda G, niter: G.copy())

    assert graphs.omega(graph, niter=1, nrand=2) == pytest.approx(0.0)
