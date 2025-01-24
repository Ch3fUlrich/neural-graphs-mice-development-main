"""Graphs module"""

import warnings
from multiprocessing import TimeoutError

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from networkx.algorithms.smallworld import lattice_reference, random_reference

from ngmd import timeseries


def adjacency_from_similarity(sim_mat, connect_thresh=0.5, binarize=False):
    """Adjacency matrix from similarity matrix

    Parameters
    ----------
    sim_mat : array-like
        Square similarity matrix
    connect_thresh : float, optional
        Connectivity threshold, by default 0.5. Entries in the adjacency matrix
        are set to zero if the absolute value of the corresponding entry in the
        similarity matrix is less than `connect_thresh`
    binarize : bool, optional
        Binarize the adjacency matrix, by default False. In a binarized
        adjacency matrix, nodes are either connected with a weight of 1 or
        disconnected (weight 0).

    Returns
    -------
    ndarray
        A square adjacency matrix. The connection weights are taken from the
        similarity matrix, if `binarize == False`. Otherwise, the connection
        weights are all 1

    Raises
    ------
    ValueError
        If the similarity matrix is no square
    """
    n, m = np.shape(sim_mat)
    if n != m:
        raise ValueError("Similarity matrix should be square")

    # Remove entries below threshold and delete the diagonal values
    # (self-connections)
    good_entries = sim_mat > connect_thresh
    W = np.multiply(sim_mat - np.multiply(np.identity(n), sim_mat), good_entries)

    # Binarize adjacency matrix if required
    if binarize:
        W = (W > 0).astype(float)

    return W


def largest_cc(G_nx):
    """Largest connected subgraph

    Parameters
    ----------
    G_nx : `networkx` graph
        The queried graph

    Returns
    -------
    `networkx` graph
        The largest connected component of `G_nx`
    """
    return G_nx.subgraph(max(nx.connected_components(G_nx), key=len)).copy()


def omega(G, niter=5, nrand=5, **kwargs):
    """Returns the small-world coefficient (omega) of a graph

    The small-world coefficient of a graph G is:

    omega = Lr/L - C/Cl

    where C and L are respectively the average clustering coefficient and
    average shortest path length of G. Lr is the average shortest path length
    of an equivalent random graph and Cl is the average clustering coefficient
    of an equivalent lattice graph.

    The small-world coefficient (omega) measures how much G is like a lattice
    or a random graph. Negative values mean G is similar to a lattice whereas
    positive values mean G is a random graph.
    Values close to 0 mean that G has small-world characteristics.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    niter: integer (optional, default=5)
        Approximate number of rewiring per edge to compute the equivalent
        lattice graphs. The corresponding rewiring number the equivalent random
        graphs is 2*niter.
    nrand: integer (optional, default=5)
        Number of random graphs generated to compute the maximal clustering
        coefficient (Cr) and average shortest path length (Lr).
    **kwargs : keyword arguments
        Additional arguments passed to the `joblib.Parallel` class

    Returns
    -------
    omega : float
        The small-world coefficient (omega)

    Notes
    -----
    The implementation is adapted from the algorithm by Telesford et al. [1]_.

    This is a reimplemented version of the NetworkX function using joblib to
    parallelize the computation of the random and lattice reference graphs.

    References
    ----------
    .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).
           "The Ubiquity of Small-World Networks".
           Brain Connectivity. 1 (0038): 367-75.  PMC 3604768. PMID 22432451.
           doi:10.1089/brain.2011.0038.
    """

    # Calculate clustering coefficient and average shortest path length for
    # original graph
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)

    # Define number of iterations for random and lattice reference graphs
    niter_lattice_reference = niter
    niter_random_reference = niter * 2

    # Define loop function
    def loop_func():
        # Generate random graph
        Gr = random_reference(G, niter=niter_random_reference)
        Lr = nx.average_shortest_path_length(Gr)

        # Generate lattice graph
        Gl = lattice_reference(G, niter=niter_lattice_reference)

        # Replace old clustering coefficient, if clustering is higher in
        # generated lattice reference
        Cl = nx.average_clustering(Gl)

        return Lr, Cl

    n_jobs = kwargs.pop("n_jobs", 1)
    if n_jobs > 1:
        try:
            # Run loop function in parallel
            results = Parallel(n_jobs=n_jobs, **kwargs)(
                delayed(loop_func)() for _ in range(nrand)
            )
        except TimeoutError:
            # If the parallel computation times out, return np.nan
            warnings.warn("Computation timed out. Returning np.nan.")
            return np.nan
    else:
        # Run loop function in serial
        results = [loop_func() for _ in range(nrand)]

    # Get maximum clustering coefficient and mean shortest path length from
    # the graphs generated in the loop function
    Cl = np.max([C] + [r[1] for r in results])
    Lr = np.mean([r[0] for r in results])

    # Calculate omega
    omega = (Lr / L) - (C / Cl)

    return omega


def remove_small_components(G, min_cc_size):
    G_pruned = G.copy()
    for component in list(nx.connected_components(G_pruned)):
        if len(component) < min_cc_size:
            for node in component:
                G_pruned.remove_node(node)
    return G_pruned


def subsample_graph(G, subsample_size):
    """Subsample a graph

    Parameters
    ----------
    G : `networkx` graph
        The graph to subsample
    subsample_size : int
        Number of nodes to subsample

    Returns
    -------
    `networkx` graph
        The subsampled graph
    """
    subsample_size = min(subsample_size, len(G))
    subsample_nodes = np.random.choice(G.nodes, size=subsample_size, replace=False)
    G_sub = G.subgraph(subsample_nodes).copy()
    return G_sub


class SimilarityGraph:
    """Similarity graph

    Parameters
    ----------
    similarity_matrix : array-like
        Square similarity matrix. Should in principle have only positive,
        finite values. Even if there are negative values, they will be
        converted to positive values via their absolute values. NaN values are
        converted to 0 and inf values are converted to very large numbers.
    connect_thresh : float, optional
        Connectivity threshold, by default None. If None, the connectivity
        threshold is set to the mean similarity plus two standard deviations

    Attributes
    ----------
    similarity_matrix : array-like
        Square similarity matrix
    connect_thresh : float
        Connectivity threshold
    W : array-like
        Weighted adjacency matrix. The non-zero weights correspond to the
        similarity values in the corresponding positions in the similarity
        matrix
    G_pygsp : `pygsp` Graph object
        A `pygsp` graph constructed from the adjacency matrix in `self.W`
    G_nx: `networkx` graph
        A networkx graph constructed from the adjacency matrix in `self.W`

    Raises
    ------
    ValueError
        If the similarity matrix is not square

    Notes
    -----
    One may reset the connectivity threshold after construction of the
    `SimilarityGraph` object (call it `sg`) by following the template
    `sg.connect_thresh = new_value`. In doing so, the adjacency matrix, and
    the pygsp and networkx graphs are automatically recomputed.
    """

    def __init__(self, similarity_matrix, connect_thresh: float = None) -> None:
        self.similarity_matrix = self._format_similarity_matrix(similarity_matrix)

        # Parse connectivity threshold
        if connect_thresh is None:
            valid_values = np.extract(
                1 - np.eye(len(self.similarity_matrix)), self.similarity_matrix
            )
            valid_values = valid_values[valid_values != 0]
            mu = np.mean(valid_values)
            std = np.std(valid_values)
            connect_thresh = mu + 2 * std

        self.connect_thresh = connect_thresh

    def _format_similarity_matrix(self, similarity_matrix):
        # Turn np.nan entries into 0 and np.inf into very large numbers
        out = np.nan_to_num(similarity_matrix)
        # Make sure all similarity values positive
        out = np.abs(out)
        return out

    @property
    def connect_thresh(self):
        return self._connect_thresh

    @connect_thresh.setter
    def connect_thresh(self, value):
        self._connect_thresh = value

        # Compute adjacency matrix
        self.W = adjacency_from_similarity(
            self.similarity_matrix, connect_thresh=self._connect_thresh
        )
        # Generate networkx graph
        self.G_nx = nx.from_numpy_array(self.W)


class NeuralActivityGraph(SimilarityGraph):
    """Neural activity graph

    Parameters
    ----------
    activity_traces : array-like, optional
        N-by-T array representing T samples from N activity trace time-series.
        Each of the time-series is supposed to represent a neuron. The activity
        trace values could be anything that could serve as a proxy for neuronal
        activity, e.g. calcium fluorescence values. Either `activity_traces`
        or `corr_mat` should be provided.
    corr_mat : array-like, optional
        N-by-N matrix made of pairwise correlations between neurons. Either
        `activity_traces` or `corr_mat` should be provided.
    connect_thresh : float, optional
        Absolute correlation level above which to consider two neurons as
        connected. By default, it takes the value of the mean correlation plus
        two standard deviations.

    Attributes
    ----------
    activity_traces : array-like
        N-by-T array representing T samples from N activity trace time-series.
        Each of the time-series is supposed to represent a neuron. It is None,
        if not provided by the user
    corr_mat : array-like
        N-by-N matrix made of pairwise correlations between neurons
    connect_thresh : float
        Connectivity threshold
    W : array-like
        Weighted adjacency matrix. The non-zero weights correspond to the
        correlation values in the corresponding positions in the correlation
        matrix
    G_nx: `networkx` graph
        A networkx graph constructed from the adjacency matrix in `self.W`

    Raises
    ------
    ValueError
        If neither `activity_traces` nor `corr_mat` are provided

    Notes
    -----
    One may reset the connectivity threshold after construction of the
    `NeuralActivityGraph` object (call it `nag`) by following the template
    `nag.connect_thresh = new_value`. In doing so, the adjacency matrix, and
    the networkx graphs are automatically recomputed.
    """

    def __init__(self, activity_traces=None, corr_mat=None, connect_thresh=None):
        # Parse activity traces and correlation matrix
        self.activity_traces = activity_traces
        if (corr_mat is None) and (self.activity_traces is not None):
            self.corr_mat = timeseries.compute_correlations(self.activity_traces)
        elif corr_mat is not None:
            self.corr_mat = corr_mat
        else:
            raise ValueError(
                "Either one of `activity_traces` or " "`corr_mat` should be provided"
            )
        super().__init__(similarity_matrix=self.corr_mat, connect_thresh=connect_thresh)
