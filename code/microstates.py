"""
Functions to segment EEG data into microstates based on 3 methods: modified
K-Means, AAHC, and TAAHC.

Some code taken from https://github.com/wmvanvliet/mne_microstates [modified
    K-Means]
Some code inspired by https://github.com/Frederic-vW/eeg_microstates [AAHC]
"""

import logging

import numpy as np
from scipy.stats import zscore

from data_utils import corr_vectors, get_gfp_peaks


def segment_microstates(
    data,
    method="mod_kmeans",
    n_states=4,
    use_gfp=True,
    normalize=False,
    return_polarity=False,
    **kwargs,
):
    """
    Segment a continuous signal into microstates using one of the three methods:
    - modified K-Means: stochastic clustering method, instead of recomputing
        cluster centers using average, it computes largest eigenvector
    - AAHC: Atomize and Agglomerative Hierarchical Clustering; bottom-up
        approach, initialised with each topomap as a cluster, and number of
        clusters is reduced in each step by re-assigning the worst cluster
        (computed by GEV) to remaining clusters
    - TAAHC: Topographic AAHC; same principle, instead of GEV for atomisation,
        the worst cluster is determined using Pearson correlation

    :param data: data to find the microstates in, channels x samples
    :type data: np.ndarray
    :param method: method to use, `mod_kmeans`, `AAHC`, `TAAHC`
    :type method: str
    :param n_states: number of states to find
    :type n_states: int
    :param use_gfp: whether to use GFP peaks to find microstates or whole data
    :type use_gfp: bool
    :param normalize: whether to z-score the data
    :type normalize: bool
    :param return_polarity: whether to return the polarity of the activation
    :type return_polarity: bool
    :**kwargs:
        - n_inits: number of random initialisations to use for algorithm of
            modified K-Means
        - max_iter: the maximum number of iterations to perform in the modified
            K-Means
        - thresh: threshold for convergence of the modified K-Means
        - min_peak_dist: minimal peak distance in GFP peaks
        - smoothing: smoothing window type for GFP curve
        - smoothing_window: smoothing window length for GFP curve
    :return: microstate maps, dummy segmentation (maximum activation), polarity
        (if `return_polarity` == True), global explained variance for whole
        timeseries, global explained variance for GFP peaks
    :rtype: (np.ndarray, np.ndarray, np.ndarray, float, float)
    """
    if normalize:
        data = zscore(data, axis=1)

    if use_gfp:
        (peaks, gfp_curve) = get_gfp_peaks(
            data,
            min_peak_dist=kwargs.pop("min_peak_dist", 2),
            smoothing=kwargs.pop("smoothing", None),
            smoothing_window=kwargs.pop("smoothing_window", 100),
        )
    else:
        peaks = np.arange(data.shape[1])
        gfp_curve = np.std(data, axis=0)

    # cache this value for later
    gfp_sum_sq = np.sum(gfp_curve ** 2)
    peaks_sum_sq = np.sum(data[:, peaks].std(axis=0) ** 2)

    if method == "mod_kmeans":
        n_inits = kwargs.pop("n_inits", 10)
        max_iter = kwargs.pop("max_iter", 1000)
        thresh = kwargs.pop("thresh", 1e-6)
        logging.debug(
            f"Using modified K-Means for finding {n_states} microstates, using "
            f"{n_inits} random initialisations..."
        )
        # several runs of the k-means algorithm, keep track of the best
        # segmentation
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

            # compare across iterations using global explained variance (GEV) of
            # the found microstates.
            gev = sum((gfp_curve * map_corr) ** 2) / gfp_sum_sq
            gev_gfp = (
                sum((data[:, peaks].std(axis=0) * gfp_corr) ** 2) / peaks_sum_sq
            )
            logging.debug(f"GEV of found microstates: {gev}")
            if gev > best_gev:
                best_gev, best_maps, best_segmentation = gev, maps, segmentation
                best_gfp_gev = gev_gfp
                best_polarity = np.sign(np.choose(segmentation, activation))

    elif method in ["AAHC", "TAAHC"]:
        logging.debug(f"Using {method} for finding {n_states} microstates")
        maps = _aahc(data.T, peaks, gfp_curve, n_states, atomisation=method)
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)
        map_corr = corr_vectors(data, maps[segmentation].T)
        gfp_corr = corr_vectors(data[:, peaks], maps[segmentation[peaks]].T)
        best_gev = sum((gfp_curve * map_corr) ** 2) / gfp_sum_sq
        best_gfp_gev = (
            sum((data[:, peaks].std(axis=0) * gfp_corr) ** 2) / peaks_sum_sq
        )
        best_maps, best_segmentation = maps, segmentation
        best_polarity = np.sign(np.choose(segmentation, activation))

    else:
        raise ValueError(f"Unknown method for microstates: {method}")

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


def _aahc(data_orig, gfp_peaks, gfp_curve, n_states, atomisation="AAHC"):
    """
    Atomize and Agglomerative Hierarchical Clustering. Possible also
    Topographic AAHC.

    Inspired by https://github.com/Frederic-vW/eeg_microstates
    """

    def extract_topomap(maps, k):
        """
        Extract kth topomap and return the remainder, along with the extracted
        topomap.

        :param maps: sequence of topomaps, channels x samples
        :type maps: np.ndarray
        :param k: index of topomap to extract
        :type k: int
        """
        v = maps[k, :]
        maps_ = np.vstack((maps[:k, :], maps[k + 1 :, :]))
        return maps_, v

    def extract_item(A, k):
        """
        Extract kth item from list/array and return the remainder, along with
        the extracted item.

        :param A: sequence to extract from
        :type A: list|np.ndarray
        :param k: index of item to extract
        :type k: int
        """
        a = A[k]
        A_ = A[:k] + A[k + 1 :]
        return A_, a

    assert atomisation in ["AAHC", "TAAHC"]
    n_ch = data_orig.shape[0]
    gfp_sum_sq = np.sum(gfp_curve ** 2)

    # initialize clusters = all maps are clusters
    maps = data_orig[gfp_peaks, :]
    cluster_data = data_orig[gfp_peaks, :]
    n_maps = maps.shape[0]

    # cluster indices w.r.t original size
    Ci = [[k] for k in range(n_maps)]

    # main loop: atomize + agglomerate
    while n_maps > n_states:
        # correlations of the data sequence with each cluster
        m_x, s_x = data_orig.mean(axis=1, keepdims=True), data_orig.std(axis=1)
        m_y, s_y = maps.mean(axis=1, keepdims=True), maps.std(axis=1)
        s_xy = 1.0 * n_ch * np.outer(s_x, s_y)
        C = np.dot(data_orig - m_x, np.transpose(maps - m_y)) / s_xy

        # microstate sequence, ignore polarity
        L = np.argmax(C ** 2, axis=1)

        atomisation_ = np.zeros(n_maps)
        for k in range(n_maps):
            r = L == k
            if atomisation == "AAHC":
                # GEV
                atomisation_[k] = (
                    np.sum(gfp_curve[r] ** 2 * C[r, k] ** 2) / gfp_sum_sq
                )
            elif atomisation == "TAAHC":
                # correlation
                atomisation_[k] = np.sum(C[r, k] ** 2)

        # merge cluster with the minimum GEV
        imin = np.argmin(atomisation_)

        maps, _ = extract_topomap(maps, imin)
        Ci, reC = extract_item(Ci, imin)
        # indices of updated clusters
        re_cluster = []
        for k in reC:  # map index to re-assign
            c = cluster_data[k, :]
            m_x, s_x = maps.mean(axis=1, keepdims=True), maps.std(axis=1)
            m_y, s_y = c.mean(), c.std()
            s_xy = 1.0 * n_ch * s_x * s_y
            C = np.dot(maps - m_x, c - m_y) / s_xy
            inew = np.argmax(C ** 2)  # ignore polarity
            re_cluster.append(inew)
            Ci[inew].append(k)
        n_maps = len(Ci)
        re_cluster = list(set(re_cluster))

        # re-clustering by eigenvector method
        for i in re_cluster:
            idx = Ci[i]
            Vt = cluster_data[idx, :]
            Sk = np.dot(Vt.T, Vt)
            evals, evecs = np.linalg.eig(Sk)
            c = evecs[:, np.argmax(np.abs(evals))]
            c = np.real(c)
            maps[i] = c / np.sqrt(np.sum(c ** 2))

    return maps


def _mod_kmeans(
    data,
    n_states=4,
    max_iter=1000,
    thresh=1e-6,
):
    """
    The modified K-means clustering algorithm.

    Code by Marijn van Vliet <w.m.vanvliet@gmail.com>
    https://github.com/wmvanvliet/mne_microstates
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
