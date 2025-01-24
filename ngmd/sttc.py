import numpy as np
import torch
from torch import FloatTensor
from torch.nn.functional import conv2d


def convolve2d(img, kernel, stride=1, padding=0):
    """Convolve a 2D image with a 2D kernel.

    Parameters
    ----------
    signal : np.ndarray
        2D image to convolve with. Implied shape is (height, width, channels).
    kernel : np.ndarray
        2D kernel to convolve with. Implied shape is (height, width).
    stride : int, optional
        Stride of the convolution, by default 1.

    Returns
    -------
    torch.Tensor
        Convolved signal.
    """
    # Turn numpy array into float tensor of shape (1, n_channels, height, width)
    img = FloatTensor(img).unsqueeze(0).unsqueeze(0)

    # Define filter from kernel
    filter = FloatTensor(kernel).unsqueeze(0).unsqueeze(0)

    # Apply filter, making sure the output shape is the same as the input shape
    with torch.no_grad():
        convolved_img = conv2d(img, filter, stride=stride, padding=padding)

    # Return the convolved image back as a numpy array of shape (height, width, channels)
    return convolved_img.squeeze(0).squeeze(0).numpy()


def tile_spike_trains(spike_matrix, dt_in_samples):
    # Create a ±dt boolean mask to tile the spike train
    tile_width = 2 * dt_in_samples + 1
    tiling_mask = np.array([True] * tile_width).reshape(1, -1)

    # Convolve the spike train with the tiling mask to get the tiling
    tiling = convolve2d(
        spike_matrix, tiling_mask, stride=1, padding=(0, int(tile_width / 2))
    )
    tiling = tiling.astype(bool)

    return tiling


def proportions_of_matched_spikes(spike_matrix, tiled_spike_matrix):
    """Compute matrix of proportions of matched spikes between spike trains

    Parameters
    ----------
    spike_matrix : array-like
        An N-by-T array of numbers representing N binary spike trains with T
        samples each
    tiled_spike_matrix : array-like
        An N-by-T array of numbers representing N binary spike trains with T
        samples each, tiled with a synchronicity window

    Returns
    -------
    array-like
        An N-by-N matrix containing pairwise proportions of matched spikes
        for the spike trains in `spike_matrix`. If A is the ROI of the first row in
        `spike_matrix`, then the row `[0, :]` in the output matrix will contain the
        proportion of spikes in A that are matched with the spikes in each of the other
        ROIs. For example, if B is the ROI of the second row in `spike_matrix`, then the entry at `[0, 1]` represents P_A_B, the proportion of spikes in A that are matched with the spikes in B.
    """
    num_spikes = np.sum(spike_matrix.astype(float), axis=1)

    # Mask the points where num_spikes is zero to avoid division by zero.
    num_spikes = np.ma.masked_equal(num_spikes, 0)

    #  Weigh the spikes by the inverse of the number of spikes in each ROI
    weighted_spike_matrix = spike_matrix.astype(float) / num_spikes[:, np.newaxis]

    # Compute proportions via masked matrix multiplication
    P = np.ma.dot(
        tiled_spike_matrix.astype(float), weighted_spike_matrix.T, strict=True
    ).T

    # Fill masked values with np.nan
    P = P.filled(np.nan)

    return P


def fraction_of_time_covered(tiled_spike_matrix):
    """
    Compute the proportion of the total recording time covered (or tiled) by the spikes of each ROI in the spike matrix.

    Parameters
    ----------
    spike_train : array-like
        A binary spike train
    dt_in_samples : int
        The number of samples to tile the spike train

    Returns
    -------
    float
        The proportion of the total recording time covered by the tiled spikes
    """
    _, num_samples = np.shape(tiled_spike_matrix)
    T = np.sum(tiled_spike_matrix, axis=1).astype(float) / num_samples
    return T


def matrix_spike_time_tiling_coefficient(spike_matrix, sample_rate, dt):
    """
    Compute Spike Time Tiling Coefficients (STTCs) for an entire spike
    matrix. Uses numba to speed up computations.

    Parameters
    ----------
    spike_matrix : array-like
        An N-by-T array of numbers representing N binary spike trains with T
        samples each
    sample_rate : float
        Sampling rate (Hz) of the spike trains
    dt : float
        The synchronicity window (in seconds) used for STTC computation

    Returns
    -------
    array-like
        An N-by-N matrix containing pairwise STTC values for the spike trains
        in `spike_matrix`
    """
    # Make tiled spike matrix
    dt_in_samples = int(dt * sample_rate)
    tiled_spikes = tile_spike_trains(spike_matrix, dt_in_samples)

    # Compute vector of fractions of time covered by the spikes (Ts)
    T_vector = fraction_of_time_covered(tiled_spikes)

    # Compute matrix of proportions of matched spikes (Ps)
    P_matrix = proportions_of_matched_spikes(spike_matrix, tiled_spikes)

    # Get mask of entries where P_matrix is np.nan
    nan_mask = np.isnan(P_matrix)

    # Compute the denominator matrix containing all terms 1 - P_A * T_B
    denominators = 1 - P_matrix * T_vector[np.newaxis, :]

    # Mask the entries where the denominator is zero
    denominators = np.ma.masked_equal(denominators, 0)

    # Compute the numerator matrix containing all terms P_A - T_B
    numerators = P_matrix - T_vector[np.newaxis, :]

    # Compute the matrix containing the terms P_A - T_B / 1 - P_A * T_B
    terms_matrix = numerators / denominators

    # Fill masked values with 1
    terms_matrix = terms_matrix.filled(1)

    # Fill values in nan_mask with np.nan. We have to do this step
    # separately because all the entries with NaNs in terms_matrix
    # have also been masked and filled with ones. Here we distinguish
    # the entries that should be filled with ones from those that
    # should be filled with NaNs.
    terms_matrix[nan_mask] = np.nan

    # Average upper and lower triangular parts of the matrix to get the final STTC
    # values
    sttc_mat = (terms_matrix + terms_matrix.T) / 2

    return sttc_mat


def test_compute_sttc():
    """Test the compute_sttc function"""
    spike_matrix = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    sttc_mat = matrix_spike_time_tiling_coefficient(spike_matrix, 1, 1)
    correct_values = np.array(
        [
            [1.0, 0.7, -0.33333333, np.nan],
            [0.7, 1.0, 0.54347826, np.nan],
            [-0.33333333, 0.54347826, 1.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    return np.testing.assert_equal(sttc_mat, correct_values)
