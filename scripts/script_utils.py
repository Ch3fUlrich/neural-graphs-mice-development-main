"""Utilities for the scripts"""

from ngmd import graphs


def mat_is_valid(sim_mat, subsample_size):
    """Check if the similarity matrix is valid

    Parameters
    ----------
    sim_mat : np.ndarray
        Similarity matrix
    subsample_size : int
        Number of nodes to subsample

    Returns
    -------
    bool
        True if the similarity matrix is valid, False otherwise
    """
    if sim_mat is None:
        return False
    elif len(sim_mat) < subsample_size:
        return False
    else:
        return True


def subsample_and_compute(G_nx, fun, subsample_size, num_subsample_trials):
    """Subsample the graph and compute a function

    Parameters
    ----------
    G_nx : nx.Graph
        Networkx graph
    fun : function
        Function to compute
    subsample_size : int
        Number of nodes to subsample
    num_subsample_trials : int
        Number of subsample trials

    Returns
    -------
    list
        List of computed values over the subsampling trials
    """
    values = []
    if subsample_size is not None:
        for _ in range(num_subsample_trials):
            G_nx_sub = graphs.subsample_graph(G_nx, subsample_size)  # TODO
            values.append(fun(G_nx_sub))
    else:
        values.append(fun(G_nx))

    return values
