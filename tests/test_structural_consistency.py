import networkx as nx
import numpy as np
import pytest

from ngmd import structural_consistency as sc


def test_get_perturbed_edges_uses_selection_fraction(monkeypatch):
    graph = nx.path_graph(5)
    monkeypatch.setattr(sc, "sample", lambda population, k: list(population)[:k])

    edges, count = sc.get_perturbed_edges(graph, p_selection=0.5)

    assert count == 2
    assert edges == [(0, 1), (1, 2)]


def test_adjacency_perturbation_helpers():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    delta = sc.get_delta_A(adjacency, [(0, 1)])
    remaining = sc.get_A_R(adjacency, delta)

    np.testing.assert_array_equal(delta, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))
    np.testing.assert_array_equal(remaining, np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]))


def test_eigendecomposition_rounds_eigenvalues_and_quadratic_form_normalizes():
    matrix = np.diag([1.00004, 2.0])

    eigenvalues, eigenvectors = sc.eigendecomposition(matrix, n_decimals_lambda=3)

    np.testing.assert_array_equal(eigenvalues, np.array([1.0, 2.0]))
    np.testing.assert_allclose(eigenvectors @ eigenvectors.T, np.eye(2))
    assert sc.normalized_quadratic_form(np.array([2.0, 0.0]), matrix) == pytest.approx(
        1.00004
    )


def test_approximation_matrix_handles_non_degenerate_and_degenerate_cases():
    delta = np.array([[0.0, 1.0], [1.0, 0.0]])
    eigenvectors = np.eye(2)

    non_degenerate = sc.compute_approximation_matrix(
        np.array([1.0, 2.0]),
        eigenvectors,
        delta,
    )
    degenerate = sc.compute_approximation_matrix(
        np.array([1.0, 1.0]),
        eigenvectors,
        delta,
    )

    assert non_degenerate.shape == delta.shape
    assert degenerate.shape == delta.shape
    np.testing.assert_allclose(non_degenerate, np.diag([1.0, 2.0]))


def test_rank_non_observed_links_and_common_link_count():
    scores = np.array(
        [
            [0.0, 10.0, 2.0],
            [10.0, 0.0, 9.0],
            [2.0, 9.0, 0.0],
        ]
    )
    observed = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

    ranked = sc.rank_non_observed_links(scores.copy(), observed, num_perturbed_edges=1)

    assert ranked == [(1, 2)]
    assert sc.get_num_common_links(ranked, [(2, 1)]) == 1


def test_structural_consistency_returns_zero_when_no_edges_are_selected():
    graph = nx.path_graph(4)

    assert sc.structural_consistency(graph, p_selection=0.0) == 0


def test_structural_consistency_returns_fraction(monkeypatch):
    graph = nx.path_graph(4)
    monkeypatch.setattr(sc, "get_perturbed_edges", lambda G, p_selection: ([(1, 2)], 1))

    value = sc.structural_consistency(graph, p_selection=0.5)

    assert 0.0 <= value <= 1.0
