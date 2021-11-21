"""
Surrogate class for single subject EEG recording.
"""

from copy import deepcopy
from functools import partial

import numpy as np
from eeg_recording import SingleSubjectRecording

DEFAULT_SEED = None


class SurrogateRecording(SingleSubjectRecording):
    """
    Surrogate class for EEG recordings
    """

    @classmethod
    def from_data(cls, subject_data):
        assert isinstance(subject_data, SingleSubjectRecording)
        surrs = cls(
            subject_data.subject_id, subject_data._data, subject_data.attrs
        )
        surrs._orig_data = deepcopy(surrs._data)
        return surrs

    def __str__(self):
        return super().__str__().replace("EEG", "Surrogate")

    def construct_surrogates(self, surrogate_type, univariate=False, **kwargs):
        """
        Construct single 1D surrogate column-wise from the signal.

        :param surrogate_type: type of the surrogate to construct, implemented:
            - shuffle
            - FT
            - AAFT
            - IAAFT
        :type surrogate_type: str
        :param univariate: if True, each column will be seeded independently
            (not preserving any kind of relationship between columns); if False
            will use multivariate construction, i.e. one seed for all
            realizations in columns, hence will preserve relationships
        :type univariate: bool
        :**kwargs: potential arguments for surrogate creation
        """
        # copy original data to _data for processing
        self._data = deepcopy(self._orig_data)

        # seed setting
        seed = kwargs.pop(
            "seed",
            None
            if univariate
            else np.random.randint(low=0, high=np.iinfo(np.uint32).max),
        )

        def get_surr(ts, surr_type, seed, **kwargs):
            # normalise
            ts = ts.T
            mean = ts.mean(axis=0)
            std = ts.std(axis=0)
            ts_ = (ts - mean) / std
            # compute surrogate
            if surr_type == "shuffle":
                surr = get_single_shuffle_surrogate(ts_, seed=seed)
            elif surr_type == "FT":
                surr = get_single_FT_surrogate(ts_, seed=seed)
            elif surr_type == "AAFT":
                surr = get_single_AAFT_surrogate(ts_, seed=seed)
            elif surr_type == "IAAFT":
                surr = get_single_IAAFT_surrogate(ts_, seed=seed, **kwargs)
            else:
                raise ValueError(f"Unknown surrogate type {surr_type}")
            # denormalise
            surr = (surr * std) + mean
            return surr.T

        # apply function to _data
        self._data.apply_function(
            partial(get_surr, surr_type=surrogate_type, seed=seed, **kwargs)
        )


def get_single_shuffle_surrogate(ts, seed=DEFAULT_SEED):
    """
    Return single 1D shuffle surrogate. Timeseries is cut into `cut_points`
    pieces at random and then shuffled. If `cut_points` is None, will cut each
    point, hence whole timeseries is shuffled.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D shuffle surrogate of timeseries
    :rtype: np.ndarray
    """
    np.random.seed(seed)
    cut_points = ts.shape[0]
    # generate random partition points without replacement
    partion_points = np.sort(
        np.random.choice(ts.shape[0], cut_points, replace=False)
    )

    def split_permute_concat(x, split_points):
        """
        Helper that splits, permutes and concats the timeseries.
        """
        return np.concatenate(
            np.random.permutation(np.split(x, split_points, axis=0))
        )

    current_permutation = split_permute_concat(ts, partion_points)
    # assert we actually permute the timeseries, useful when using only one
    # cutting point, i.e. two partitions so they are forced to swap
    while np.all(current_permutation == ts):
        current_permutation = split_permute_concat(ts, partion_points)
    return current_permutation


def get_single_FT_surrogate(ts, seed=DEFAULT_SEED):
    """
    Returns single 1D Fourier transform surrogate.

    Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Farmer, J. D.
        (1992). Testing for nonlinearity in time series: the method of
        surrogate data. Physica D: Nonlinear Phenomena, 58(1-4), 77-94.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D FT surrogate of timeseries
    :rtype: np.ndarray
    """
    np.random.seed(seed)
    if ts.ndim == 1:
        ts = ts[:, np.newaxis]
    xf = np.fft.rfft(ts, axis=0)
    angle = np.random.uniform(0, 2 * np.pi, (xf.shape[0],))[:, np.newaxis]
    # set the slowest frequency to zero, i.e. not to be randomised
    angle[0] = 0

    cxf = xf * np.exp(1j * angle)

    return np.fft.irfft(cxf, n=ts.shape[0], axis=0).squeeze()


def get_single_AAFT_surrogate(ts, seed=DEFAULT_SEED):
    """
    Returns single 1D amplitude-adjusted Fourier transform surrogate.

    Schreiber, T., & Schmitz, A. (2000). Surrogate time series. Physica D:
        Nonlinear Phenomena, 142(3-4), 346-382.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D AAFT surrogate of timeseries
    :rtype: np.ndarray
    """
    # create Gaussian data
    if ts.ndim == 1:
        ts = ts[:, np.newaxis]
    gaussian = np.broadcast_to(
        np.random.randn(ts.shape[0])[:, np.newaxis], ts.shape
    )
    gaussian = np.sort(gaussian, axis=0)
    # rescale data to Gaussian distribution
    ranks = ts.argsort(axis=0).argsort(axis=0)
    rescaled_data = np.zeros_like(ts)
    for i in range(ts.shape[1]):
        rescaled_data[:, i] = gaussian[ranks[:, i], i]
    # do phase randomization
    phase_randomized_data = get_single_FT_surrogate(rescaled_data, seed=seed)
    if phase_randomized_data.ndim == 1:
        phase_randomized_data = phase_randomized_data[:, np.newaxis]
    # rescale back to amplitude distribution of original data
    sorted_original = ts.copy()
    sorted_original.sort(axis=0)
    ranks = phase_randomized_data.argsort(axis=0).argsort(axis=0)

    for i in range(ts.shape[1]):
        rescaled_data[:, i] = sorted_original[ranks[:, i], i]

    return rescaled_data.squeeze()


def get_single_IAAFT_surrogate(ts, n_iterations=1000, seed=DEFAULT_SEED):
    """
    Returns single 1D iteratively refined amplitude-adjusted Fourier transform
    surrogate. A set of AAFT surrogates is iteratively refined to produce a
    closer match of both amplitude distribution and power spectrum of surrogate
    and original data.

    Schreiber, T., & Schmitz, A. (2000). Surrogate time series. Physica D:
        Nonlinear Phenomena, 142(3-4), 346-382.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param n_iterations: number of iterations of the procedure
    :type n_iterations: int
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D IAAFT surrogate of timeseries
    :rtype: np.ndarray
    """
    if ts.ndim == 1:
        ts = ts[:, np.newaxis]
    # FT of original data
    xf = np.fft.rfft(ts, axis=0)
    # FT amplitudes
    xf_amps = np.abs(xf)
    sorted_original = ts.copy()
    sorted_original.sort(axis=0)

    # starting point of iterative procedure
    R = get_single_AAFT_surrogate(ts, seed=seed)
    if R.ndim == 1:
        R = R[:, np.newaxis]
    # iterate: `R` is the surrogate with "true" amplitudes and `s` is the
    # surrogate with "true" spectrum
    for _ in range(n_iterations):
        # get Fourier phases of R surrogate
        r_fft = np.fft.rfft(R, axis=0)
        r_phases = r_fft / np.abs(r_fft)
        # transform back, replacing the actual amplitudes by the desired
        # ones, but keeping the phases exp(i*phase(i))
        s = np.fft.irfft(xf_amps * r_phases, n=ts.shape[0], axis=0)
        #  rescale to desired amplitude distribution
        ranks = s.argsort(axis=0).argsort(axis=0)
        for j in range(R.shape[1]):
            R[:, j] = sorted_original[ranks[:, j], j]

    return R.squeeze()
