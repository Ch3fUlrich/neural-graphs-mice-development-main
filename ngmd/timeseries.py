"""Time-series processing module"""

import itertools

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata

from ngmd import sttc
from ngmd.utils import fill_sym_mat_from_triu


def compute_correlations(array, **kwargs):
    """Compute Pearson correlation matrix from array of time-series

    Parameters
    ----------
    array : array-like
        An N-by-T array of numbers representing N time-series with T samples
        each

    Returns
    -------
    array-like
        An N-by-N matrix containing pairwise Pearson correlation coefficients
        for the time series in `array`
    """
    corr_mat = np.corrcoef(array, **kwargs)
    return corr_mat


def make_time_shuffle_surrogates(binary_traces, max_shift=None):
    """Generate surrogate traces by rolling each neuron's binary trace by a
    random amount of time

    Parameters
    ----------
    binary_traces : array-like
        An N-by-T array of numbers representing N binary spike trains with T
        samples each
    max_shift : int, optional
        Maximum number of time steps to shift each binary trace, by default None. If
        None, the maximum shift is set to the number of time steps

    Returns
    -------
    array-like
        An N-by-T array of numbers representing N surrogate binary spike trains
        with T samples each
    """
    num_neurons, num_time_steps = binary_traces.shape
    surrogate_traces = np.zeros_like(binary_traces)
    # Draw `num_neurons` random time shifts
    if max_shift is None:
        max_shift = num_time_steps
    time_shifts = np.random.randint(0, max_shift, num_neurons)
    # For each neuron, shift its binary trace by the corresponding time shift
    for i in range(num_neurons):
        surrogate_traces[i, :] = np.roll(binary_traces[i, :], time_shifts[i])
    return surrogate_traces


def make_pulse_shuffle_surrogates(binary_traces):
    """Generate surrogate traces by shuffling the pulses among neurons while keeping
    the pulse times intact

    Parameters
    ----------
    binary_traces : array-like
        An N-by-T array of numbers representing N binary spike trains with T
        samples each

    Returns
    -------
    array-like
        An N-by-T array of numbers representing N surrogate binary spike trains
        with T samples each
    """
    num_neurons, num_time_steps = binary_traces.shape
    surrogate_traces = np.zeros_like(binary_traces)
    # Find time step windows with overlapping pulses. We do this to avoid breaking
    # up pulses when shuffling the roi assignments
    has_pulse = np.any(binary_traces, axis=0)
    # Find time steps where pulse windows begin and end
    transitions = np.diff(has_pulse.astype(int), prepend=0, append=0)
    beginnings = np.where(transitions == 1)[0]
    endings = np.where(transitions == -1)[0]
    # Build surrogate traces by shuffling the roi ids of the pulse windows
    # column-wise
    for i, (b, e) in enumerate(zip(beginnings, endings)):
        roi_assignments = np.random.permutation(num_neurons)
        surrogate_traces[:, b:e] = binary_traces[roi_assignments, b:e]
    return surrogate_traces


def make_poisson_surrogate_traces(binary_traces):
    """Generate surrogate traces by drawing Poisson distributed pulse intervals with the
    same pulse rate as the original trace

    The pulse widths of the surrogate trace are drawn from the pulse widths of the
    original trace. The pulse times are drawn from the Poisson distributed pulse
    intervals.

    Parameters
    ----------
    binary_traces : array-like
        An N-by-T array of numbers representing N binary spike trains with T
        samples each

    Returns
    -------
    array-like
        An N-by-T array of numbers representing N surrogate binary traces
        with T samples each
    """
    num_neurons, num_time_steps = binary_traces.shape
    surrogate_traces = np.zeros_like(binary_traces)
    for i in range(num_neurons):
        # Gather statistics from the ROI trace
        results = gather_binary_trace_statistics(binary_traces[i, :], 1)
        n_pulses = results["n_pulses"]
        duration = results["duration"]
        pulse_widths = results["pulse_widths"]

        # Draw Poisson distributed pulse intervals with the same pulse rate as the
        # original trace
        pulse_rate = n_pulses / duration
        pulse_intervals = np.random.poisson(1 / pulse_rate, n_pulses)
        pulse_times = np.cumsum(pulse_intervals).astype(int)

        # Generate the Poisson surrogate trace
        surrogate_trace = np.zeros((num_time_steps,))
        for j, width in enumerate(pulse_widths):
            surrogate_trace[pulse_times[j] : pulse_times[j] + int(width)] = 1
        surrogate_traces[i, :] = surrogate_trace

    return surrogate_traces


def make_surrogate_traces(binary_traces, method="time_shuffle", **kwargs):
    """Generate surrogate traces using a specified method

    Parameters
    ----------
    binary_traces : array-like
        An N-by-T array of numbers representing N binary spike trains with T
        samples each
    method : str, optional
        Method to use for generating surrogate traces, by default "time_shuffle". Possible values are: "time_shuffle", "pulse_shuffle", "time_and_pulse_shuffle", "poisson"
    **kwargs
        Additional keyword arguments to pass to the surrogate generation function

    Returns
    -------
    array-like
        An N-by-T array of numbers representing N surrogate binary spike trains
        with T samples each

    Raises
    ------
    ValueError
        If an invalid method is specified

    See Also
    --------
    make_time_shuffle_surrogates
    make_pulse_shuffle_surrogates
    make_poisson_surrogate_traces
    """
    possible_methods = [
        "time_shuffle",
        "pulse_shuffle",
        "time_and_pulse_shuffle",
        "poisson",
    ]
    if method == "time_shuffle":
        return make_time_shuffle_surrogates(binary_traces, **kwargs)
    elif method == "pulse_shuffle":
        return make_pulse_shuffle_surrogates(binary_traces)
    elif method == "time_and_pulse_shuffle":
        surrogate_traces = make_time_shuffle_surrogates(binary_traces, **kwargs)
        surrogate_traces = make_pulse_shuffle_surrogates(surrogate_traces)
        return surrogate_traces
    elif method == "poisson":
        return make_poisson_surrogate_traces(binary_traces)
    else:
        raise ValueError(f"Invalid method: {method}. Choose from: {possible_methods}")


def compile_surrogate_sttc_values(binary_traces, sample_rate, **params):
    """Compile surrogate STTC values from multiple trials

    Parameters
    ----------
    binary_traces : array-like
        An N-by-T array of numbers representing N binary spike trains with T
        samples each
    sample_rate : float
        Sampling rate (Hz) of the binary traces
    params : dict, optional
        A dictionary containing the following parameters:
        - method: Method to use for generating surrogate traces, by default "pulse_shuffle"
        - num_trials: Number of trials to run, by default 3
        - n_jobs: Number of parallel jobs to run, by default 1
        - dt: Synchronicity window (in seconds) used for STTC computation, by default 1

    Returns
    -------
    array-like
        A num_trials-by-num_pairs array containing surrogate STTC values
    """
    method = params.get("method", "pulse_shuffle")
    num_trials = params.get("num_trials", 3)
    n_jobs = params.get("n_jobs", 1)
    dt = params.get("dt", 1)

    # Define surrogate STTC computation function to be run each trial
    def get_surrogate_sttc_values(trial):
        surrogate_traces = make_surrogate_traces(binary_traces, method=method)
        surrogate_sttc_mat = sttc.matrix_spike_time_tiling_coefficient(
            surrogate_traces, sample_rate=sample_rate, dt=dt
        )
        num_rois, _ = binary_traces.shape
        triu_values = surrogate_sttc_mat[np.triu_indices(num_rois, 1)]
        return triu_values

    # Run surrogate STTC computation in parallel
    surrogate_values_collection = Parallel(n_jobs=n_jobs)(
        delayed(get_surrogate_sttc_values)(trial) for trial in range(num_trials)
    )

    # Stack surrogate STTC values as a num_trials-by-num_pairs array
    surrogate_values_collection = np.vstack(surrogate_values_collection)

    return surrogate_values_collection


def compute_percentiles_matrix(original_matrix, surrogate_triu_values):
    """Compute a matrix with percentiles for the values of the input matrix

    Parameters
    ----------
    original_matrix : array-like
        An N-by-N matrix containing the values for which to compute percentiles
    surrogate_triu_values : array-like
        An array of shape (num_trials, num_pairs) or (num_trials  * num_pairs, )
        containing surrogate STTC values. See `compile_surrogate_sttc_values`, for example

    Returns
    -------
    array-like
        An N-by-N matrix containing percentiles for the original STTC matrix
    """
    #  Extract upper triangular part of the matrix, excluding the main diagonal
    num_rows = original_matrix.shape[0]
    original_triu_values = original_matrix[np.triu_indices(num_rows, 1)]

    # Append the original values to the surrogate values
    all_values = np.concatenate((original_triu_values, surrogate_triu_values.flatten()))

    # Remove NaN values
    all_values = all_values[~np.isnan(all_values)]

    # Compute percentile scores from ranked data
    percentile_values = (
        100 * (rankdata(all_values) / len(all_values))[: len(original_triu_values)]
    )

    # Fill symmetric percentiles matrix with the computed upper triangular values
    percentiles_mat = fill_sym_mat_from_triu(num_rows, percentile_values, diag=np.nan)
    return percentiles_mat


def parse_transitions(trace):
    """Parse the 0->1 and 1->0 transitions in a binary trace

    Parameters
    ----------
    trace : array-like
        A binary trace of a neuron's activity

    Returns
    -------
    tuple
        A tuple containing the indices of the positive (0->1) and negative (1->0)
        transitions in the binary trace
    """
    # Compute the 0-1, 1-0 transition sequence
    transitions = np.diff(trace)

    # Gather the positive and negative transitions
    positive_transitions = np.where(transitions > 0)[0]
    neg_transitions = np.where(transitions < 0)[0]

    # If the length of the negative transitions is lesser than the
    # length of the positive transitions, it means that the last pulse is
    # still active. Add a negative transition at the end of the trace to
    # account for this.
    if len(neg_transitions) < len(positive_transitions):
        neg_transitions = np.append(neg_transitions, len(trace))

    # If the length of the negative transitions is greater than the length
    # of the positive transitions, it means that the first pulse started
    # active. Add a positive transition at the beginning of the trace to
    # account for this.
    if len(neg_transitions) > len(positive_transitions):
        positive_transitions = np.append(0, positive_transitions)

    return positive_transitions, neg_transitions


def gather_binary_trace_statistics(trace, sample_rate):
    """Gather statistics from a binary trace

    Parameters
    ----------
    trace : array-like
        A binary trace of a neuron's activity
    sample_rate : float
        Sampling rate (Hz) of the binary trace

    Returns
    -------
    dict
        A dictionary containing the following statistics:
        - duration: Duration of the trace in seconds
        - n_pulses: Number of pulses in the trace
        - pulse_widths: List of pulse widths in seconds
        - pulse_intervals: List of pulse intervals in seconds
        - active_fraction: Fraction of time when there are active pulses
    """
    # Get duration of the trace in seconds
    n_samples = len(trace)
    duration = n_samples / sample_rate

    # Parse the transitions
    positive_transitions, neg_transitions = parse_transitions(trace)

    # Count the number of pulses
    n_pulses = len(positive_transitions)

    # Gather the pulse_widths (s)
    pulse_widths = list((neg_transitions - positive_transitions) / sample_rate)

    # Gather the pulse_intervals (s). These are the differences between the
    # positive transitions, minus the pulse widths.
    pulse_intervals = list(np.diff(positive_transitions) / sample_rate)
    pulse_intervals = [x - y for x, y in zip(pulse_intervals, pulse_widths[:-1])]

    # Gather the fraction of time when there are active pulses
    active_time = sum(pulse_widths)
    active_fraction = active_time / duration

    return {
        "duration": duration,
        "n_pulses": n_pulses,
        "pulse_widths": pulse_widths,
        "pulse_intervals": pulse_intervals,
        "active_fraction": active_fraction,
    }
