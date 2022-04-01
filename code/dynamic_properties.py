"""
Set of dynamic properties of microstate sequences.

Inspired by https://github.com/Frederic-vW/eeg_microstates
"""
import logging

import numpy as np
from hurst import compute_Hc
from scipy.stats import chi2
from tqdm import tqdm

LOG_FUNC = {True: np.log2, False: np.log}
DEFAULT_ALPHA = 0.01

# Basic distributions et al.


def empirical_distribution(sequence, n_states=4):
    """
    Compute empirical distribution of a sequence given n_states.

    :param sequence: input sequence, should contain symbols starting from 0
    :type sequence: np.ndarray
    :param n_states: number of allowed states in the sequence, symbols then
        should be in [0, n_states-1]
    :type n_states: int
    :return: empirical distribution of a sequence
    :rtype: np.ndarray
    """
    p = np.zeros(n_states)
    for i in range(sequence.shape[0]):
        p[sequence[i]] += 1.0
    p /= sequence.shape[0]
    assert np.abs(p.sum() - 1.0) < 1e-6, p.sum()
    return p


def empirical_trans_mat(sequence, n_states=4):
    """
    Compute empirical transition matrix of a sequence given n_states.

    :param sequence: input sequence, should contain symbols starting from 0
    :type sequence: np.ndarray
    :param n_states: number of allowed states in the sequence, symbols then
        should be in [0, n_states-1]
    :type n_states: int
    :return: empirical transition matrix of a sequence as [from, to]
    :rtype: np.ndarray
    """
    prob_matrix = np.zeros((n_states, n_states))
    for from_, to_ in zip(sequence, sequence[1:]):
        prob_matrix[from_, to_] += 1
    transition_mat = prob_matrix / np.nansum(prob_matrix, axis=1, keepdims=True)
    return transition_mat


def equilibrium_distribution(trans_mat):
    """
    Compute equilibrium (stationary) distribution from a given transition matrix
    as `lambda = 1 - (left) eigenvector`

    :param trans_mat: empirical transition matrix as [from, to]
    :type trans_mat: np.ndarray
    :return: equilibrium distribution given the transition matrix
    :rtype: np.ndarray
    """
    evals, evecs = np.linalg.eig(trans_mat.transpose())
    # index of maximum eigenvalue
    i = np.where(np.isclose(evals, 1.0, atol=1e-6))[0][0]
    # make eigenvec. to max. eigenval. non-negative
    p_eq = np.abs(evecs[:, i])
    # normalized eigenvec. to max. eigenval.
    p_eq /= p_eq.sum()
    return p_eq


def mixing_time(trans_mat):
    """
    Compute relaxation time, which is an inverse of spectral gap.

    :param trans_mat: empirical transition matrix as [from, to]
    :type trans_mat: np.ndarray
    :return: mixing (relaxation) time
    :rtype: float
    """
    ev = np.linalg.eigvals(trans_mat)
    ev = np.real(ev)
    ev.sort()
    ev2 = np.flipud(ev)
    # spectral gap
    sg = ev2[0] - ev2[1]
    T_mix = 1.0 / sg
    return T_mix


# Entropies


def max_entropy(n_states, log2=True):
    """
    Maximum Shannon entropy given number of possible states.

    :param n_states: number of allowed states in the sequence
    :type n_states: int
    :param log2: whether to use base 2 logarithm [entropy in bits] or natural
        logarithm [entropy in nats]
    :type log2: bool
    :return: maximum Shannon entropy
    :rtype: float
    """
    h_max = LOG_FUNC[log2](float(n_states))
    return h_max


def H_1(sequence, n_states, log2=True):
    """
    Shannon entropy of the symbolic sequence with n_states symbols.

    :param sequence: input sequence, should contain symbols starting from 0
    :type sequence: np.ndarray
    :param n_states: number of allowed states in the sequence, symbols then
        should be in [0, n_states-1]
    :type n_states: int
    :param log2: whether to use base 2 logarithm [entropy in bits] or natural
        logarithm [entropy in nats]
    :type log2: bool
    :return: Shannon entropy of a sequence
    :rtype: float
    """
    p = empirical_distribution(sequence=sequence, n_states=n_states)
    h = -np.sum(p[p > 0] * LOG_FUNC[log2](p[p > 0]))
    return h


def H_2(x, y, n_states, log2=True):
    """
    Joint Shannon entropy of the symbolic sequences X, Y with n_states symbols.

    :param x: first input sequence, should contain symbols starting from 0
    :type x: np.ndarray
    :param y: second input sequence, should contain symbols starting from 0
    :type y: np.ndarray
    :param n_states: number of allowed states in the sequence, symbols then
        should be in [0, n_states-1]
    :type n_states: int
    :param log2: whether to use base 2 logarithm [entropy in bits] or natural
        logarithm [entropy in nats]
    :type log2: bool
    :return: joint Shannon entropy of x and y
    :rtype: float
    """
    if len(x) != len(y):
        logging.warning("Sequences of different lengths, using shorter...")
    n = min([len(x), len(y)])
    p = np.zeros((n_states, n_states))
    for t in range(n):
        p[x[t], y[t]] += 1.0
    p /= n
    h = -np.sum(p[p > 0] * LOG_FUNC[log2](p[p > 0]))
    return h


def H_k(sequence, n_states, k, log2=True):
    """
    Shannon joint entropy of sequence k-history.

    :param sequence: input sequence, should contain symbols starting from 0
    :type sequence: np.ndarray
    :param n_states: number of allowed states in the sequence, symbols then
        should be in [0, n_states-1]
    :type n_states: int
    :param k: history w.r.t which H is evaluated
    :type k: int
    :param log2: whether to use base 2 logarithm [entropy in bits] or natural
        logarithm [entropy in nats]
    :type log2: bool
    :return: Shannon entropy of a sequence
    :rtype: float
    """
    N = len(sequence)
    f = np.zeros(tuple(k * [n_states]))
    for t in range(N - k):
        f[tuple(sequence[t : t + k])] += 1.0
    f /= N - k  # normalize distribution
    hk = -np.sum(f[f > 0] * LOG_FUNC[log2](f[f > 0]))
    return hk


def excess_entropy_rate(sequence, n_states, kmax, log2=True):
    """
    Compute entropy rate and excess entropy over histories of a sequence up to
    kmax k-history.
    - y = ax+b: line fit to joint entropy for range of histories k
    - a = entropy rate (slope)
    - b = excess entropy (intersect.)

    :param sequence: input sequence, should contain symbols starting from 0
    :type sequence: np.ndarray
    :param n_states: number of allowed states in the sequence, symbols then
        should be in [0, n_states-1]
    :type n_states: int
    :param kmax: maximum history w.r.t which H is evaluated
    :type kmax: int
    :param log2: whether to use base 2 logarithm [entropy in bits] or natural
        logarithm [entropy in nats]
    :type log2: bool
    :return: Shannon entropy rate and excess entropy over kmax k-history of a
        sequence
    :rtype: (float, float)
    """
    h_ = np.zeros(kmax)
    for k in range(kmax):
        h_[k] = H_k(sequence, n_states, k + 1, log2=log2)
    ks = np.arange(1, kmax + 1)
    a, b = np.polyfit(ks, h_, 1)
    return a, b


def markov_chain_entropy_rate(distribution, trans_mat, log2=True):
    """
    Theoretical entropy rate of Markov chain as sum_i sum_j p_i T_ij log(T_ij)

    :param distribution: distribution of symbols in Markov chain
    :type distribution: np.ndarray
    :param trans_mat: transition matrix in Markov chain
    :type trans_mat: np.ndarray
    :param log2: whether to use base 2 logarithm [entropy in bits] or natural
        logarithm [entropy in nats]
    :type log2: bool
    :return: entropy rate of a Markov chain given its distribution and
        transition matrix
    :rtype: float
    """
    h = 0.0
    for i, j in np.ndindex(trans_mat.shape):
        if trans_mat[i, j] > 0:
            h -= (
                distribution[i]
                * trans_mat[i, j]
                * LOG_FUNC[log2](trans_mat[i, j])
            )
    return h


def lagged_mutual_information(
    sequence, n_states, max_lag, log2=True, pbar=True
):
    """
    Time lagged mutual information of a sequence with n_states symbols.

    :param sequence: input sequence, should contain symbols starting from 0
    :type sequence: np.ndarray
    :param n_states: number of allowed states in the sequence, symbols then
        should be in [0, n_states-1]
    :type n_states: int
    :param max_lag: maximum time lag, in samples
    :type max_lag: int
    :param log2: whether to use base 2 logarithm [entropy in bits] or natural
        logarithm [entropy in nats]
    :type log2: bool
    :param pbar: whether to use progress bar (useful in notebooks, bad in
        scripts with potentially multiple progress bars)
    :type pbar: bool
    :return: time-lagged mutual information, per lag
    :rtype: np.ndarray
    """
    mi = np.zeros(max_lag)
    if pbar:
        range_ = tqdm(range(max_lag))
    else:
        range_ = range(max_lag)
    for lag in range_:
        nmax = sequence.shape[0] - lag
        h1 = H_1(sequence[:nmax], n_states, log2)
        h2 = H_1(sequence[lag : lag + nmax], n_states, log2)
        h12 = H_2(sequence[:nmax], sequence[lag : lag + nmax], n_states, log2)
        mi[lag] = h1 + h2 - h12
    return mi


def find_1st_aif_peak(lagged_mi, sampling_freq):
    """
    Fins 1st peak in the lagged mutual information in ms.

    :param lagged_mi: lagged mutual information timeseries
    :type lagged_mi: np.ndarray
    :param sampling_freq: sampling frequency of original sequence, in Hz
    :type sampling_freq: float
    :return: index of 1st peak of AIF, it's time in ms
    :rtype: (int, float)
    """
    # sampling interval [ms]
    dt = 1000.0 / sampling_freq
    # smoothed version of MI
    mi_smooth = np.convolve(lagged_mi, np.ones(3) / 3.0, mode="same")
    mx0 = 8

    def locmax(x):
        """
        Find local maxima in x using 1st order derivatives.
        """
        dx = np.diff(x)  # discrete 1st derivative
        zc = np.diff(np.sign(dx))  # zero-crossings of dx
        m = 1 + np.where(zc == -2)[0]  # indices of local max.
        return m

    try:
        jmax = mx0 + locmax(mi_smooth[mx0:])[0]
        mx_mi = dt * jmax
    except IndexError:
        jmax, mx_mi = np.nan, np.nan

    return jmax, mx_mi


# Markovian stuff


def test_markovianity_nth_order(
    sequence,
    n_states,
    order,
    log2=True,
    alpha=DEFAULT_ALPHA,
    detailed=False,
):
    """
    Test n-th order Markovianity of a sequence with n_states symbols. Works for
    orders 0, 1, and 2.
    H0: nth-order MC <=>
        0th order: p(X[t]), p(X[t+1])
        1st order: p(X[t+1] | X[t]) = p(X[t+1] | X[t], X[t-1])
        2nd order: p(X[t+1] | X[t], X[t-1]) = p(X[t+1] | X[t], X[t-1], X[t-2])

    Reference: Kullback, S., Kupperman, M., & Ku, H. H. (1962). Tests for
        contingency tables and Markov chains. Technometrics, 4(4), 573-608.

    :param sequence: input sequence, should contain symbols starting from 0
    :type sequence: np.ndarray
    :param n_states: number of allowed states in the sequence, symbols then
        should be in [0, n_states-1]
    :type n_states: int
    :param order: order of Markovianity to test, in [0, 1, 2]
    :type order: int
    :param log2: whether to use base 2 logarithm [entropy in bits] or natural
        logarithm [entropy in nats]
    :type log2: bool
    :param alpha: significance level
    :type: alpha: float
    :param detailed: if True, returns also T statistic and df
    :type detailed: bool
    :return: p-value of Chi2 test for independence, whether to reject H0 (True)
        or not (False), if detailed also T statistic and degrees of freedom
    :rtype: (float, bool), or (float, bool, float, int)
    """
    assert order in [0, 1, 2]
    n = len(sequence)
    f_full = np.zeros([n_states] * (order + 2))
    f_partial1 = np.zeros([n_states] * (order + 1))
    f_partial2 = np.zeros_like(f_partial1)
    if order > 0:
        f_small = np.zeros([n_states] * order)

    def _get_sequence_indices(x, t, order):
        indices = []
        for i in range(order + 2):
            indices.append(x[t + i])
        return tuple(indices)

    for t in range(n - (order + 1)):
        indices = _get_sequence_indices(sequence, t, order)
        f_full[indices] += 1.0
        f_partial1[indices[:-1]] += 1.0
        f_partial2[indices[1:]] += 1.0
        if order > 0:
            f_small[indices[1:-1]] += 1.0
    T = 0.0  # statistic
    for indices in np.ndindex(f_full.shape):
        f = f_full[indices] * f_partial1[indices[:-1]] * f_partial2[indices[1:]]
        if order > 0:
            f *= f_small[indices[1:-1]]
        if f > 0:
            num_ = f_full[indices]
            if order == 0:
                num_ *= n
            else:
                num_ *= f_small[indices[1:-1]]
            den_ = f_partial1[indices[:-1]] * f_partial2[indices[1:]]
            T += f_full[indices] * LOG_FUNC[log2](num_ / den_)

    T *= 2.0
    df = (n_states - 1.0) * (n_states - 1.0) * np.power(n_states, order)
    pval = chi2.sf(T, df, loc=0, scale=1)
    logging.debug(f"p-value: {pval:.2e} | t: {T:.3f} | df: {df:.1f}")
    if detailed:
        return pval, pval < alpha, T, df
    else:
        return pval, pval < alpha


def test_stationarity_conditional_homogeneity(
    sequence,
    n_states,
    block_size,
    alpha=DEFAULT_ALPHA,
    detailed=False,
):
    """
    Test for conditional homogeneity of non-overlapping blocks of block_size.

    :param sequence: input sequence, should contain symbols starting from 0
    :type sequence: np.ndarray
    :param n_states: number of allowed states in the sequence, symbols then
        should be in [0, n_states-1]
    :type n_states: int
    :param block_size: block size for splitting of a sequence
    :type block_size: int
    :param alpha: significance level
    :type: alpha: float
    :param detailed: if True, returns also T statistic and df
    :type detailed: bool
    :return: p-value of Chi2 test for conditional homogeneity, whether to reject
        H0 (True) or not (False), if detailed also T statistic and degrees of
        freedom
    :rtype: (float, bool), or (float, bool, float, int)
    """
    n = len(sequence)
    num_blocks = int(np.floor(float(n) / float(block_size)))  # number of blocks
    logging.debug(
        f"Split data in r = {num_blocks} blocks of length {block_size}."
    )
    f_ijk = np.zeros((num_blocks, n_states, n_states))
    f_ij = np.zeros((num_blocks, n_states))
    f_jk = np.zeros((n_states, n_states))
    f_i = np.zeros(num_blocks)
    f_j = np.zeros(n_states)

    # calculate f_ijk (time / block dep. transition matrix)
    for i in range(num_blocks):  # block index
        for ii in range(block_size - 1):  # pos. inside the current block
            j = sequence[i * block_size + ii]
            k = sequence[i * block_size + ii + 1]
            f_ijk[i, j, k] += 1.0
            f_ij[i, j] += 1.0
            f_jk[j, k] += 1.0
            f_i[i] += 1.0
            f_j[j] += 1.0

    # conditional homogeneity (Markovian stationarity)
    T = 0.0
    for i, j, k in np.ndindex(f_ijk.shape):
        # conditional homogeneity
        f = f_ijk[i, j, k] * f_j[j] * f_ij[i, j] * f_jk[j, k]
        if f > 0:
            num_ = f_ijk[i, j, k] * f_j[j]
            den_ = f_ij[i, j] * f_jk[j, k]
            T += f_ijk[i, j, k] * np.log(num_ / den_)
    T *= 2.0
    df = (num_blocks - 1) * (n_states - 1) * n_states

    pval = chi2.sf(T, df, loc=0, scale=1)
    logging.debug(f"p-value: {pval:.2e} | t: {T:.3f} | df: {df:.1f}")
    if detailed:
        return pval, pval < alpha, T, df
    else:
        return pval, pval < alpha


def estimate_hurst(
    sequence, min_window, max_window, sampling_freq, detailed=False
):
    """
    Estimate Hurst exponent using `hurst` python package. Optionally return also
    data for plotting.

    :param sequence: sequence of symbols from which to estimate Hurst exponent
    :type sequence: np.ndarray
    :param min_window: minimum window for estimation, in seconds
    :type min_window: float
    :param max_window: minimum window for estimation, in seconds
    :type max_window: float
    :param sampling_freq: samping frequency of the sequence, in Hz
    :type sampling_freq: float
    :param detailed: if True, returns also intercept after log-log Hurst fit and
        data for plotting, if False returns only estimate of Hurst exponent
        (slope)
    :type detailed: bool
    :return: estimate of the Hurst exponent, optionally also intercept and
        plotting data with [time_intervals, R/S ratios]
    :rtype: float or (float, float, list[np.ndarray])
    """
    H, c, data = compute_Hc(
        sequence,
        kind="change",
        simplified=False,
        min_window=int(min_window * sampling_freq),
        max_window=int(max_window * sampling_freq),
    )
    if detailed:
        return H, c, data
    else:
        return H
