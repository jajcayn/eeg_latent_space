"""
Set of clustering scores for microstates.
"""

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import euclidean_distances


def pascual_marqui_variance_test(gev, no_states, n_channels):
    """
    Compute variance test for microstate segmentation. Lower is better.

    Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995). Segmentation of
    brain electrical activity into microstates: model estimation and validation.
    IEEE Transactions on Biomedical Engineering, 42(7), 658-665 ~ eq. (20).

    :param gev: global `unexplained` (i.e. 1 - explained variance) variance by
        microstate decomposition
    :type gev: float
    :param no_states: number of canonical states for decomposition
    :type no_states: int
    :param n_channels: number of channels in EEG recording
    :type n_channels: int
    :return: P-M variance test, lower is better
    :rtype: float
    """
    return gev * np.power(
        (1.0 / (n_channels - 1)) * (n_channels - 1 - no_states), -2
    )


def davies_bouldin_test(eeg_data, segmentation):
    """
    Compute Davies-Bouldin clustering score. Lower is better.

    :param eeg_data: original EEG data, as time x channels
    :type eeg_data: np.ndarray
    :param segmentation: segmentation time series
    :type segmentation: np.ndarray
    :return: Davies-Bouldin clustering score, lower is better
    :rtype: float
    """
    return davies_bouldin_score(eeg_data, segmentation)


def dunn_test(eeg_data, segmentation):
    """
    Compute Dunn clustering score. Higher is better.
    Taken from https://github.com/jqmviegas/jqm_cvi/blob/master/jqmcvi/base.py

    :param eeg_data: original EEG data, as time x channels
    :type eeg_data: np.ndarray
    :param segmentation: segmentation time series
    :type segmentation: np.ndarray
    :return: Dunn clustering score, higher is better
    :rtype: float
    """
    distances = euclidean_distances(eeg_data)
    ks = np.sort(np.unique(segmentation))

    deltas = np.ones([len(ks), len(ks)]) * 1000000
    big_deltas = np.zeros([len(ks), 1])

    l_range = list(range(0, len(ks)))

    def delta_fast(ck, cl, distances):
        values = distances[np.where(ck)][:, np.where(cl)]
        values = values[np.nonzero(values)]

        return np.min(values)

    def big_delta_fast(ci, distances):
        values = distances[np.where(ci)][:, np.where(ci)]
        return np.max(values)

    for k in l_range:
        for l in l_range[0:k] + l_range[k + 1 :]:
            deltas[k, l] = delta_fast(
                (segmentation == ks[k]), (segmentation == ks[l]), distances
            )

        big_deltas[k] = big_delta_fast((segmentation == ks[k]), distances)

    di = np.min(deltas) / np.max(big_deltas)
    return di


def silhouette_test(eeg_data, segmentation):
    """
    Compute Silhouette score as a silhouette coefficient of all samples. Result
    in [1, -1], with 1 being the best.

    :param eeg_data: original EEG data, as time x channels
    :type eeg_data: np.ndarray
    :param segmentation: segmentation time series
    :type segmentation: np.ndarray
    :return: Silhouette clustering score, 1 is best, -1 is worst
    :rtype: float
    """
    return silhouette_score(eeg_data, segmentation)


def calinski_harabasz_test(eeg_data, segmentation):
    """
    Compute Calinksi-Harabasz score, also called Variance Ratio criterion.
    Higher is better.

    :param eeg_data: original EEG data, as time x channels
    :type eeg_data: np.ndarray
    :param segmentation: segmentation time series
    :type segmentation: np.ndarray
    :return: Calinski-Harabasz score, higher is better
    :rtype: float
    """
    return calinski_harabasz_score(eeg_data, segmentation)
