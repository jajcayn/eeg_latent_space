"""
Functions to segment EEG into microstates. Based on the Microsegment toolbox
for EEGlab, written by Andreas Trier Poulsen [1]_.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>

References
----------
.. [1]  Poulsen, A. T., Pedroni, A., Langer, N., &  Hansen, L. K. (2018).
        Microstate EEGlab toolbox: An introductionary guide. bioRxiv.


Taken from https://github.com/wmvanvliet/mne_microstates

Little changes / additions by me.
"""

import logging
from data_utils import corr_vectors

import numpy as np
from scipy.signal import convolve, find_peaks, get_window
from scipy.stats import zscore


def get_gfp_peaks(data, min_peak_dist=2, smoothing=None, smoothing_window=100):
    """
    Compute GFP peaks.

    :param data: data for GFP peaks, channels x samples
    :type data: np.ndarray
    :param min_peak_dist: minimum distance between two peaks
    :type min_peak_dist: int
    :param smoothing: smoothing window if some, None means to smoothing
    :type smoothing: str|None
    :param smoothing_window: window for smoothing, in samples
    :type smoothing_window: int
    :return: GFP peaks and GFP curve
    :rtype: (list, np.ndarray)
    """
    gfp_curve = np.std(data, axis=0)
    if smoothing is not None:
        gfp_curve = convolve(
            gfp_curve,
            get_window(smoothing, Nx=smoothing_window),
        )
    gfp_peaks, _ = find_peaks(gfp_curve, distance=min_peak_dist)

    return gfp_peaks, gfp_curve


def segment(
    data,
    n_states=4,
    use_gfp=True,
    n_inits=10,
    max_iter=1000,
    thresh=1e-6,
    normalize=False,
    return_polarity=False,
    **kwargs,
):
    """
    Segment a continuous signal into microstates.

    Peaks in the global field power (GFP) are used to find microstates, using a
    modified K-means algorithm. Several runs of the modified K-means algorithm
    are performed, using different random initializations. The run that
    resulted in the best segmentation, as measured by global explained variance
    (GEV), is used.

    Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
    Additions: Nikola Jajcay
    Code: https://github.com/wmvanvliet/mne_microstates

    :param data: data to find the microstates in, channels x samples
    :type data: np.ndarray
    :param n_states: number of states to find
    :type n_states: int
    :param use_gfp: whether to use GFP peaks to find microstates or whole data
    :type use_gfp: bool
    :param n_inits: number of random initialisations to use for algorithm
    :type n_inits: int
    :param max_iter: the maximum number of iterations to perform in the
        microstate algorithm
    :type max_iter: int
    :param thresh: threshold for convergence of the microstate algorithm
    :type thresh: float
    :param normalize: whether to z-score the data
    :type normalize: bool
    :param return_polarity: whether to return the polarity of the activation
    :type return_polarity: bool
    :return: microstate maps, dummy segmentation (maximum activation), polarity
        (if `return_polarity` == True), global explained variance for whole
        timeseries, global explained variance for GFP peaks
    :rtype: (np.ndarray, np.ndarray, np.ndarray, float, float)

    References:
    Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995). Segmentation of
        brain electrical activity into microstates: model estimation and
        validation. IEEE Transactions on Biomedical Engineering, 42(7), 658-665.
    """
    logging.debug(
        "Finding %d microstates, using %d random intitializations"
        % (n_states, n_inits)
    )

    if normalize:
        data = zscore(data, axis=1)

    if use_gfp:
        (peaks, gfp_curve) = get_gfp_peaks(data, **kwargs)
    else:
        peaks = np.arange(data.shape[1])
        gfp_curve = np.std(data, axis=0)

    # Cache this value for later
    gfp_sum_sq = np.sum(gfp_curve ** 2)
    peaks_sum_sq = np.sum(data[:, peaks].std(axis=0) ** 2)

    # Do several runs of the k-means algorithm, keep track of the best
    # segmentation.
    best_gev = 0
    best_gfp_gev = 0
    best_maps = None
    best_segmentation = None
    best_polarity = None
    for _ in range(n_inits):
        maps = _mod_kmeans(
            data[:, peaks],
            n_states,
            max_iter,
            thresh,
        )
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)
        map_corr = corr_vectors(data, maps[segmentation].T)
        gfp_corr = corr_vectors(data[:, peaks], maps[segmentation[peaks]].T)
        # assigned_activations = np.choose(segmentations, activation)

        # Compare across iterations using global explained variance (GEV) of
        # the found microstates.
        gev = sum((gfp_curve * map_corr) ** 2) / gfp_sum_sq
        gev_gfp = (
            sum((data[:, peaks].std(axis=0) * gfp_corr) ** 2) / peaks_sum_sq
        )
        logging.debug("GEV of found microstates: %f" % gev)
        if gev > best_gev:
            best_gev, best_maps, best_segmentation = gev, maps, segmentation
            best_gfp_gev = gev_gfp
            best_polarity = np.sign(np.choose(segmentation, activation))

    if return_polarity:
        return (
            best_maps,
            best_segmentation,
            best_polarity,
            best_gev,
            best_gfp_gev,
        )
    else:
        return best_maps, best_segmentation, best_gev, best_gfp_gev


def _mod_kmeans(
    data,
    n_states=4,
    max_iter=1000,
    thresh=1e-6,
):
    """
    The modified K-means clustering algorithm.

    See :func:`segment` for the meaning of the parameters and return
    values.

    Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
    Code: https://github.com/wmvanvliet/mne_microstates
    """
    n_channels, n_samples = data.shape

    # Cache this value for later
    data_sum_sq = np.sum(data ** 2)

    # Select random timepoints for our initial topographic maps
    init_times = np.random.choice(n_samples, size=n_states, replace=False)
    maps = data[:, init_times].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)  # Normalize the maps

    prev_residual = np.inf
    for iteration in range(max_iter):
        # Assign each sample to the best matching microstate
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)

        # Recompute the topographic maps of the microstates, based on the
        # samples that were assigned to each state.
        for state in range(n_states):
            idx = segmentation == state
            if np.sum(idx) == 0:
                logging.warning("Some microstates are never activated")
                maps[state] = 0
                continue

            # Find largest eigenvector
            # cov = data[:, idx].dot(data[:, idx].T)
            # _, vec = eigh(cov, eigvals=(n_channels - 1, n_channels - 1))
            # maps[state] = vec.ravel()
            maps[state] = data[:, idx].dot(activation[state, idx])
            maps[state] /= np.linalg.norm(maps[state])

        # Estimate residual noise
        act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
        residual = abs(data_sum_sq - act_sum_sq)
        residual /= float(n_samples * (n_channels - 1))

        # Have we converged?
        if (prev_residual - residual) < (thresh * residual):
            logging.debug("Converged at %d iterations." % iteration)
            break

        prev_residual = residual
    else:
        logging.warning("Modified K-means algorithm failed to converge.")

    return maps
