from random import sample

import networkx as nx
import numpy as np


def get_perturbed_edges(G, p_selection):
    num_edges = len(G.edges)
    num_perturbed_edges = int(p_selection * num_edges)
    perturbed_edges = sample(list(G.edges), num_perturbed_edges)
    return perturbed_edges, num_perturbed_edges


def get_delta_A(A, perturbed_edges):
    delta_A = np.zeros_like(A)
    for edge in perturbed_edges:
        delta_A[edge] = 1
    # Make sure the perturbed adjacency matrix is symmetric
    delta_A = delta_A + delta_A.T
    return delta_A


def get_A_R(A, delta_A):
    return A - delta_A


def eigendecomposition(A, n_decimals_lambda):
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    # Round the eigenvalues to the desired number of decimals, because
    # we need to assess whether some of them are equal and so need to
    # fix any numerical imprecision
    eigenvalues = np.round(eigenvalues, decimals=n_decimals_lambda)
    return eigenvalues, eigenvectors


def normalized_quadratic_form(vector, matrix):
    # Calculate delta_lambda_k
    delta_lambda_k = vector.T @ matrix @ vector
    delta_lambda_k /= np.dot(vector, vector)
    return delta_lambda_k


def non_degenerate_case(eigenvalues_R, eigenvectors_R, delta_A):
    tilde_A = np.zeros_like(delta_A, dtype=float)
    for k in range(len(eigenvalues_R)):
        # Get the k-th eigenvalue and eigenvector from A_R
        val_k = eigenvalues_R[k]
        vec_k = eigenvectors_R[:, k]

        # Calculate delta_lambda_k
        delta_lambda_k = normalized_quadratic_form(vec_k, delta_A)

        # Increment tilde_A
        tilde_A += (val_k + delta_lambda_k) * np.outer(vec_k, vec_k)

    return tilde_A


def degenerate_case(eigenvalues_R, eigenvectors_R, delta_A):
    tilde_A = np.zeros_like(delta_A, dtype=float)
    # Get the unique eigenvalues and their multiplicities
    unique_vals, multiplicities = np.unique(eigenvalues_R, return_counts=True)
    for i, val in enumerate(unique_vals):
        if multiplicities[i] == 1:
            # Non-degenerate eigenvalue

            # Get position of the corresponding eigenvector in the
            # eigenvalues_R array
            k = np.where(eigenvalues_R == val)[0][0]

            # Get the corresponding eigenvector
            vec_k = eigenvectors_R[:, k]

            # Calculate delta_lambda_k
            delta_lambda_k = normalized_quadratic_form(vec_k, delta_A)

            # Increment tilde_A
            tilde_A += (val + delta_lambda_k) * np.outer(vec_k, vec_k)
        else:
            # Degenerate eigenvalue

            # Get the positions of the corresponding eigenvectors in
            # the eigenvalues_R array
            k = np.where(eigenvalues_R == val)[0]

            # Get the corresponding eigenvectors
            vecs_k = eigenvectors_R[:, k]

            # Form the quadratic form matrix W
            W = vecs_k.T @ delta_A @ vecs_k

            # Solve the spectral decomposition of W
            eigenvalues_W, eigenvectors_W = np.linalg.eigh(W)

            # Define a new vector basis
            new_vecs_k = vecs_k @ eigenvectors_W

            # Increment tilde_A
            for j, delta_ew in enumerate(eigenvalues_W):
                vec_k = new_vecs_k[:, j]
                tilde_A += (val + delta_ew) * np.outer(vec_k, vec_k)

    return tilde_A


def compute_approximation_matrix(eigenvalues_R, eigenvectors_R, delta_A):
    if len(eigenvalues_R) == len(np.unique(eigenvalues_R)):
        tilde_A = non_degenerate_case(eigenvalues_R, eigenvectors_R, delta_A)
    else:
        tilde_A = degenerate_case(eigenvalues_R, eigenvectors_R, delta_A)
    return tilde_A


def rank_non_observed_links(tilde_A, A_R, num_perturbed_edges):
    # Mask out the entries in tilde_A that appear in A_R
    tilde_A[A_R.astype(bool)] = -np.inf

    # Set the lower triangle to -inf, including the main diagonal, because
    # the matrix is symmetric and we don't want to count self-connections
    tilde_A[np.tril_indices_from(tilde_A)] = -np.inf

    # Get the indices of the `num_perturbed_edges` largest values in
    # tilde_A
    indices = np.argpartition(tilde_A, -num_perturbed_edges, axis=None)[
        -num_perturbed_edges:
    ]
    indices_unraveled = np.unravel_index(indices, tilde_A.shape)
    top_ranked_links = [
        (indices_unraveled[0][i], indices_unraveled[1][i])
        for i in range(num_perturbed_edges)
    ]
    return top_ranked_links


def get_num_common_links(top_ranked_links, perturbed_edges):
    num_common_links = 0
    for edge in top_ranked_links:
        for perturbed_edge in perturbed_edges:
            if (edge == perturbed_edge) or (edge == perturbed_edge[::-1]):
                num_common_links += 1
                break
    return num_common_links


def structural_consistency(G, p_selection=0.1, n_decimals_lambda=4):
    """Compute the structural consistency of a graph

    Parameters
    ----------
    G : `networkx` graph
        The queried graph
    p_selection : float, optional
        Fraction of edges to select for the perturbation set delta_E, by
        default 0.1
    n_decimals_lambda : int, optional
        Number of decimals to round the eigenvalues of A_R, the adjacency
        matrix without the perturbed edges, by default 4

    Returns
    -------
    float
        The structural consistency of the graph

    Notes
    -----
    Structural consistency is defined according to (Lü et al. 2015) [1]_.

    References
    ----------
    .. [1] Lü, L., Pan, L., Zhou, T., Zhang, Y.-C., & Stanley, H. E. (2015).
    Toward link predictability of complex networks. Proceedings of the
    National Academy of Sciences, 112(8), 2325-2330.
    https://doi.org/10.1073/pnas.1424644112
    """
    # Make sure to convert the node labels to integers from 0 to number of
    # nodes minus 1 to avoid indexing problems
    G_new = nx.convert_node_labels_to_integers(G)

    # Get the graph's adjacency matrix
    A = nx.adjacency_matrix(G_new, weight=None).toarray()

    # Step 1: select a random fraction of the edges in the graph to
    # constitute a perturbation set delta_E
    perturbed_edges, num_perturbed_edges = get_perturbed_edges(G_new, p_selection)
    if num_perturbed_edges <= 0:
        return 0

    # Define delta_A, the adjacency matrix perturbation
    delta_A = get_delta_A(A, perturbed_edges)

    # Define A_R, the perturbed adjacency matrix
    A_R = get_A_R(A, delta_A)

    # Step 2: calculate the eigenvalues and eigenvectors of A_R
    eigenvalues_R, eigenvectors_R = eigendecomposition(A_R, n_decimals_lambda)

    # Steps 3 and 4: calculate the approximation matrix tilde_A
    tilde_A = compute_approximation_matrix(eigenvalues_R, eigenvectors_R, delta_A)

    # Step 5: rank the non-observed links according to their score in
    # tilde_A
    top_ranked_links = rank_non_observed_links(tilde_A, A_R, num_perturbed_edges)

    # Step 6: calculate the structural consistency by comparing how many of
    # the top ranked links are also in the set of perturbed links
    num_common_links = get_num_common_links(top_ranked_links, perturbed_edges)
    sc = num_common_links / num_perturbed_edges
    return sc
