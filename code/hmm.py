"""
Set of helper functions for getting latent states using Hidden Markov Model.
"""
from copy import deepcopy

import numpy as np
from data_utils import corr_vectors
from hmmlearn.hmm import GaussianHMM
from microstates import get_gfp_peaks
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def segment_hmm(
    data,
    n_states=4,
    use_gfp=True,
    pca_preprocess=None,
    max_iter=200,
    normalize=False,
    return_polarity=False,
    **kwargs,
):
    """
    Segment a signal into latent space using Gaussian Hidden Markov Model.

    :param data: data to find latent states in, channels x samples
    :type data: np.ndarray
    :param n_states: number of states to find
    :type n_states: int
    :param use_gfp: whether to use GFP peaks to find microstates or whole data
    :type use_gfp: bool
    :param pca_preprocess: whether and how to use PCA to preprocess input signal
        None - do not use PCA
        int - number of PCA components to use instead of all channels
        float 0-1 - ratio of variance to keep in the data
    :type pca_preprocess: None|int|float
    :param max_iter: the maximum number of iterations to perform in the HMM
        estimation
    :type max_iter: int
    :param normalize: whether to z-score the data
    :type normalize: bool
    :param return_polarity: whether to return the polarity of the activation
    :type return_polarity: bool
    :return: microstate maps, dummy segmentation (maximum activation), polarity
        (if `return_polarity` == True), global explained variance for whole
        timeseries, global explained variance for GFP peaks
    :rtype: (np.ndarray, np.ndarray, np.ndarray, float, float)
    """

    if normalize:
        data = zscore(data, axis=1)

    if pca_preprocess:
        pca = PCA(n_components=pca_preprocess)
        train_data = pca.fit_transform(data.T).T
    else:
        train_data = deepcopy(data)

    if use_gfp:
        (peaks, gfp_curve) = get_gfp_peaks(train_data, **kwargs)
    else:
        peaks = np.arange(train_data.shape[1])
        gfp_curve = np.std(train_data, axis=0)

    gfp_sum_sq = np.sum(gfp_curve ** 2)
    peaks_sum_sq = np.sum(data[:, peaks].std(axis=0) ** 2)

    # fit HMM
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=max_iter,
        tol=1e-6,
        verbose=False,
    )
    model.fit(train_data[:, peaks].T)

    # get timeseries - full, not only peaks
    if pca_preprocess:
        _, decoding = model.decode(
            pca.fit_transform(data.T), algorithm="viterbi"
        )
    else:
        _, decoding = model.decode(data.T, algorithm="viterbi")

    # get one-hot encoding - design matrix
    oh_enc = OneHotEncoder(sparse=False)
    one_hot = oh_enc.fit_transform(decoding[:, np.newaxis])

    # get HMM maps via linear regression
    lin_reg = LinearRegression(fit_intercept=False)
    scaler = StandardScaler()
    orig_data_scaled = scaler.fit_transform(data.T)
    lin_reg.fit(one_hot, orig_data_scaled)

    maps = deepcopy(lin_reg.coef_.T)
    segmentation = deepcopy(decoding)
    activation = maps.dot(data)
    polarity = np.sign(np.choose(segmentation, activation))

    map_corr = corr_vectors(data, maps[segmentation].T)
    gfp_corr = corr_vectors(data[:, peaks], maps[segmentation[peaks]].T)
    gev = sum((gfp_curve * map_corr) ** 2) / gfp_sum_sq
    gev_gfp = sum((data[:, peaks].std(axis=0) * gfp_corr) ** 2) / peaks_sum_sq

    if return_polarity:
        return maps, segmentation, polarity, gev, gev_gfp
    else:
        return maps, segmentation, gev, gev_gfp
