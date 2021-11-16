"""
Convenience functions for data handling and basic operations.
"""

import os
from utils import DATA_ROOT
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.io import loadmat

DEFAULT_TEMPLATES = os.path.join(DATA_ROOT, "MS-templates_Koenig")


def load_Koenig_microstate_templates(n_states=4, path=DEFAULT_TEMPLATES):
    """
    Load microstate canonical maps as per Koening et al. Neuroimage, 2015.

    :param n_states: number of canonical templates to load
    :type n_states: int
    :param path: folder with templates
    :type path: str
    :return: template maps (state x channels), channel names
    :rtype: (np.ndarray, list)
    """
    assert n_states <= 6
    template_maps = loadmat(os.path.join(path, "MS_templates_Koenig.mat"))[
        "data"
    ]
    channels = pd.read_csv(os.path.join(path, "channel_info.csv"))["labels"]
    # keep only relevant maps
    template_maps = template_maps[:, :n_states, n_states - 1]
    assert template_maps.shape == (len(channels), n_states)

    return template_maps.T, channels.values.tolist()


def corr_vectors(A, B, axis=0):
    """
    Compute pairwise correlation of multiple pairs of vectors.

    Fast way to compute correlation of multiple pairs of vectors without
    computing all pairs as would with corr(A,B). Borrowed from Oli at Stack
    overflow. Note the resulting coefficients vary slightly from the ones
    obtained from corr due differences in the order of the calculations.
    (Differences are of a magnitude of 1e-9 to 1e-17 depending of the tested
    data).

    Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
    Additions: Nikola Jajcay
    Code: https://github.com/wmvanvliet/mne_microstates

    :param A: first collection of vectors
    :type A: np.ndarray
    :param B: second collection of vectors
    :type B: np.ndarray
    :param axis: axis along which to perform correlations
    :type axis: int
    :return: correlation between pairs of vector
    :rtype: np.ndarray
    """
    An = A - np.mean(A, axis=axis, keepdims=True)
    Bn = B - np.mean(B, axis=axis, keepdims=True)
    An /= np.linalg.norm(An, axis=axis, keepdims=True)
    Bn /= np.linalg.norm(Bn, axis=axis, keepdims=True)
    return np.sum(An * Bn, axis=axis)


def match_reorder_topomaps(
    maps_input,
    maps_sortby,
    return_correlation=False,
    return_attribution_only=False,
):
    """
    Match and reorder topomaps. `maps_input` will be reorderer based on
    correlations with `maps_sortby`. Disregards polarity as usual in microstates
    analyses.

    :param maps_input: maps to be reordered, no maps x channels
    :type maps_input: np.ndarray
    :param maps_sortby: reference maps for sorting, no maps x channels
    :type maps_sortby: np.ndarray
    :param return_correlation: whether to return correlations of the best
        attribution
    :type return_correlation: bool
    :param return_attribution_only: whether to return only attribution list,
        i.e. list of indices of the highest correlation, if False, will return
        reordered maps
    :type return_attribution_only: bool
    :return: best attribution or reordered maps, correlation of best attribution
        (if `return_correlation` == True)
    :rtype: np.ndarray or list[int]|n.ndarray
    """
    assert maps_input.shape == maps_sortby.shape, (
        maps_input.shape,
        maps_sortby.shape,
    )
    n_maps = maps_input.shape[0]
    best_corr_mean = -1
    best_attribution = None
    best_corr = None
    for perm in permutations(range(n_maps)):
        corr_attr = np.abs(
            corr_vectors(maps_sortby, maps_input[perm, :], axis=1)
        )
        if corr_attr.mean() > best_corr_mean:
            best_corr_mean = corr_attr.mean()
            best_corr = corr_attr
            best_attribution = perm
    to_return = (
        best_attribution
        if return_attribution_only
        else maps_input[best_attribution, :]
    )
    if return_correlation:
        return to_return, best_corr
    else:
        return to_return
