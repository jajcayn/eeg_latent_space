"""
Base class for single subject EEG recording.
"""

import logging
import os
import string
from copy import deepcopy
from functools import reduce
from itertools import groupby

import mne
import numpy as np
import pandas as pd
import scipy.stats as sts
from sklearn.decomposition import PCA, FastICA

from data_utils import (
    corr_vectors,
    get_gfp_peaks,
    load_Koenig_microstate_templates,
    match_reorder_topomaps,
)
from hmm import segment_hmm
from microstates import segment_microstates


class SingleSubjectRecording:
    """
    Base class for single subject EEG recording.
    """

    def __init__(self, subject_id, data, attributes=None):
        """
        :param subject_id: subject identifier
        :type subject_id: str
        :param data: EEG data in MNE format
        :type data: `mne.io.BaseRaw`
        :param attributes: additional attributes as a dict
        :type attributes: dict|None
        """
        self.subject_id = str(subject_id)
        assert isinstance(data, mne.io.BaseRaw)
        self._data = data
        self._data.pick_types(eeg=True)
        self.gfp_peaks = None
        self.gfp_curve = None
        # holders for latent topomaps and timeseries
        self.latent_maps = None
        self.latent_segmentation = None
        # holder for segmented timeseries statistics
        self.computed_stats = False
        self.attrs = attributes or {}

    def __str__(self):
        return (
            f"EEG data for subject {self.subject_id} with {self.data.shape[0]}"
            f" channels and {self.data.shape[1]} samples"
        )

    def __repr__(self):
        return self.__str__()

    @property
    def info(self):
        """
        Return mne info.

        :return: EEG recording info
        :rtype: dict
        """
        return self._data.info

    @property
    def data(self):
        """
        Return data as numpy array.

        :return: EEG data in mne structure, channels x time
        :rtype: np.ndarray
        """
        return self._data.get_data()

    def preprocess(self, low, high):
        """
        Preprocess data - average reference and band-pass filter.

        :param low | high: low / high frequencies for bandpass filter
        :type low | high: float
        """
        self._data.set_eeg_reference("average")
        self._data.filter(low, high)

    def gfp(self):
        """
        Compute GFP curve and peaks from EEG.
        """
        self.gfp_peaks, self.gfp_curve = get_gfp_peaks(
            self.data, min_peak_dist=2, smoothing=None, smoothing_window=100
        )

    def match_reorder_segmentation(
        self, latent_templates, template_channels, return_correlation=False
    ):
        """
        Match and reorder microstates based on template [typically group mean
        maps or Koenig's microstates templates]. Finds maximum average
        correlation among all possible attributions

        :param latent_templates: templates for sorting / sort by
        :type latent_templates: np.ndarray
        :param template_channels: list of channels in the template
        :type template_channels: list
        :param return_correlation: whether to return correlation between
            templates and latent maps
        :type return_correlation: bool
        """
        # match channels
        _, idx_input, idx_sortby = np.intersect1d(
            self.info["ch_names"], template_channels, return_indices=True
        )
        attribution, corr_w_template = match_reorder_topomaps(
            self.latent_maps[:, idx_input],
            latent_templates[:, idx_sortby],
            return_correlation=True,
            return_attribution_only=True,
        )
        self.corrs_template = corr_w_template
        # reorder latent maps
        self.latent_maps = self.latent_maps[attribution, :]
        # reorder segmentation
        new_segmentation = np.empty_like(self.latent_segmentation)
        for k, v in enumerate(attribution):
            new_segmentation[self.latent_segmentation == v] = k
        self.latent_segmentation = new_segmentation
        # recompute best polarity
        self.polarity = np.sign(
            np.choose(self.latent_segmentation, self.latent_maps.dot(self.data))
        )
        # GEV and GEV GFP should be the same
        if return_correlation:
            return corr_w_template

    def reassign_segmentation_by_midpoints(self):
        """
        Redo segmentation based by midpoints - the GFP peaks are labelled based
        on correspondence and neighbours are smooth between them.
        """
        assert self.latent_maps is not None
        if self.gfp_peaks is None:
            self.gfp()
        latent_segmentation = np.ones_like(self.gfp_curve, dtype=np.int)
        latent_segmentation *= self.latent_maps.shape[0] * 2
        for peak in self.gfp_peaks:
            # list of corr. coefs between original maps and 4 latent topomaps
            pearson = [
                np.abs(sts.pearsonr(mic, self.data[:, peak])[0])
                for mic in self.latent_maps
            ]
            # pick microstate with max corr.
            latent_segmentation[peak] = pearson.index(max(pearson))

        # midpoints between latent topomaps (temporal sense)
        peaks = self.gfp_peaks.copy()
        midpoints = [
            (peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)
        ]

        for idx in range(len(midpoints) - 1):
            # fill between two midpoints with microstate at peak
            latent_segmentation[
                midpoints[idx] : midpoints[idx + 1] + 1
            ] = latent_segmentation[peaks[idx + 1]]

        # beginning and end of ts, since these were omitted in the loop
        latent_segmentation[: midpoints[0]] = latent_segmentation[peaks[0]]
        latent_segmentation[midpoints[-1] :] = latent_segmentation[peaks[-1]]
        self.latent_segmentation = latent_segmentation

    def run_latent_kmeans(self, n_states, use_gfp=True, n_inits=50):
        """
        Run microstate segmentation. Gets canonical microstates using modified
        K-Means clustering and timeseries segmentation using the dummy rule -
        maximal activation.

        :param n_states: number of canonical microstates
        :type n_states: int
        :param n_inits: number of initialisations for the modified KMeans
            algorithm
        :type n_inits: int
        """
        (
            self.latent_maps,
            self.latent_segmentation,
            self.polarity,
            self.gev_tot,
            self.gev_gfp,
        ) = segment_microstates(
            self.data,
            method="mod_kmeans",
            n_states=n_states,
            use_gfp=use_gfp,
            n_inits=n_inits,
            return_polarity=True,
        )

    def run_latent_aahc(self, n_states, use_gfp=True, n_inits=50):
        """
        Run microstate segmentation. Gets canonical microstates using AAHC
        algorithm and timeseries segmentation using the dummy rule -
        maximal activation.

        :param n_states: number of canonical microstates
        :type n_states: int
        :param n_inits: number of initialisations for the modified KMeans
            algorithm
        :type n_inits: int
        """
        (
            self.latent_maps,
            self.latent_segmentation,
            self.polarity,
            self.gev_tot,
            self.gev_gfp,
        ) = segment_microstates(
            self.data,
            method="AAHC",
            n_states=n_states,
            use_gfp=use_gfp,
            n_inits=n_inits,
            return_polarity=True,
        )

    def run_latent_taahc(self, n_states, use_gfp=True, n_inits=50):
        """
        Run microstate segmentation. Gets canonical microstates using TAAHC
        algorithm and timeseries segmentation using the dummy rule -
        maximal activation.

        :param n_states: number of canonical microstates
        :type n_states: int
        :param n_inits: number of initialisations for the modified KMeans
            algorithm
        :type n_inits: int
        """
        (
            self.latent_maps,
            self.latent_segmentation,
            self.polarity,
            self.gev_tot,
            self.gev_gfp,
        ) = segment_microstates(
            self.data,
            method="TAAHC",
            n_states=n_states,
            use_gfp=use_gfp,
            n_inits=n_inits,
            return_polarity=True,
        )

    def _run_latent_sklearn(self, sklearn_algo, n_states, use_gfp=True):
        algo = sklearn_algo(n_components=n_states)
        # input to sklearn algos as samples x features
        if use_gfp and self.gfp_peaks is None:
            self.gfp()
            data = self.data[:, self.gfp_peaks].copy()
        else:
            data = self.data.copy()
        algo.fit(data.T)
        self.latent_maps = algo.components_.copy()
        activation = self.latent_maps.dot(self.data)
        segmentation = np.argmax(np.abs(activation), axis=0)
        self.latent_segmentation = segmentation
        self.polarity = np.sign(np.choose(segmentation, activation))
        if self.gfp_curve is None:
            self.gfp()
        gfp_sum_sq = np.sum(self.gfp_curve**2)
        peaks_sum_sq = np.sum(self.data[:, self.gfp_peaks].std(axis=0) ** 2)
        map_corr = corr_vectors(self.data, self.latent_maps[segmentation].T)
        gfp_corr = corr_vectors(
            self.data[:, self.gfp_peaks],
            self.latent_maps[segmentation[self.gfp_peaks]].T,
        )
        self.gev_tot = sum((self.gfp_curve * map_corr) ** 2) / gfp_sum_sq
        self.gev_gfp = (
            sum((self.data[:, self.gfp_peaks].std(axis=0) * gfp_corr) ** 2)
            / peaks_sum_sq
        )

    def run_latent_pca(self, n_states, use_gfp=True):
        self._run_latent_sklearn(
            sklearn_algo=PCA, n_states=n_states, use_gfp=use_gfp
        )

    def run_latent_ica(self, n_states, use_gfp=True):
        self._run_latent_sklearn(
            sklearn_algo=FastICA, n_states=n_states, use_gfp=use_gfp
        )

    def run_latent_hmm(
        self, n_states, use_gfp=True, envelope=True, pca_preprocess=None
    ):
        # HMM on envelopes
        if envelope:
            self._data.apply_hilbert(envelope=True)
        (
            self.latent_maps,
            self.latent_segmentation,
            self.polarity,
            self.gev_tot,
            self.gev_gfp,
        ) = segment_hmm(
            self.data,
            n_states=n_states,
            use_gfp=use_gfp,
            pca_preprocess=pca_preprocess,
            return_polarity=True,
        )

    def _compute_lifespan(self):
        """
        Computes average lifespan of microstates in segmented time series in ms.
        """
        assert self.latent_segmentation is not None
        consec = np.array(
            [(x, len(list(y))) for x, y in groupby(self.latent_segmentation)]
        )[1:-1]
        self.avg_lifespan = {
            ms_no: (
                consec[consec[:, 0] == ms_no].mean(axis=0)[1]
                / self.info["sfreq"]
            )
            * 1000.0
            for ms_no in np.unique(self.latent_segmentation)
        }

    def _compute_coverage(self):
        """
        Computes total coverage of microstates in segmented time series.
        """
        assert self.latent_segmentation is not None
        self.coverage = {
            ms_no: count / self.latent_segmentation.shape[0]
            for ms_no, count in zip(
                *np.unique(self.latent_segmentation, return_counts=True)
            )
        }

    def _compute_freq_of_occurrence(self):
        """
        Computes average frequency of occurrence of microstates in segmented
        time series per second.
        """
        assert self.latent_segmentation is not None
        freq_occurence = {}
        for ms_no in np.unique(self.latent_segmentation):
            idx = np.where(self.latent_segmentation[:-1] == ms_no)[0]
            count_ = np.nonzero(np.diff(self.latent_segmentation)[idx])[
                0
            ].shape[0]
            freq_occurence[ms_no] = count_ / (
                self.latent_segmentation.shape[0] / self.info["sfreq"]
            )
        self.freq_occurence = freq_occurence

    def _compute_transition_matrix(self):
        """
        Computes transition probability matrix.
        """
        assert self.latent_segmentation is not None
        prob_matrix = np.zeros(
            (self.latent_maps.shape[0], self.latent_maps.shape[0])
        )
        for from_, to_ in zip(
            self.latent_segmentation, self.latent_segmentation[1:]
        ):
            prob_matrix[from_, to_] += 1
        self.transition_mat = prob_matrix / np.nansum(
            prob_matrix, axis=1, keepdims=True
        )

    def compute_segmentation_stats(self):
        """
        Compute statistics on segmented time series, i.e. coverage, frequency of
        occurence, average lifespan and transition probability matrix.
        """
        self._compute_coverage()
        self._compute_freq_of_occurrence()
        self._compute_lifespan()
        self._compute_transition_matrix()
        self.computed_stats = True

    def save_latent(self, path, suffix=""):
        """
        Save latent maps and segmentation to npz file.
        """
        if suffix != "":
            suffix = "_" + suffix
        np.savez(
            os.path.join(path, f"{self.subject_id}{suffix}.npz"),
            latent_maps=self.latent_maps,
            channels=self.info["ch_names"],
            latent_segmentation=self.latent_segmentation,
            polarity=self.polarity,
            gev_tot=self.gev_tot,
            gev_gfp=self.gev_gfp,
        )

    def get_stats_pandas(self, write_attrs=False):
        """
        Return all segmentation statistics as pd.DataFrame for subsequent
        statistical analysis.

        :param write_attrs: whether to write attributes to dataframe
        :type write_attrs: bool
        :return: dataframe with segmented time series statistics per latent
            topomap
        :rtype: pd.DataFrame
        """
        assert self.computed_stats
        if not hasattr(self, "corrs_template"):
            self.corrs_template = [np.nan] * self.latent_maps.shape[0]
        topo_names = list(string.ascii_uppercase)[: self.latent_maps.shape[0]]
        df = pd.DataFrame(
            columns=[
                "subject_id",
                "latent map",
                "var_GFP",
                "var_total",
                "template_corr",
                "coverage",
                "occurrence",
                "lifespan",
            ]
            + [f"transition->{to_ms}" for to_ms in topo_names]
        )
        for topo_idx, topo_name in enumerate(topo_names):
            df.loc[topo_idx] = [
                self.subject_id,
                topo_name,
                self.gev_gfp,
                self.gev_tot,
                self.corrs_template[topo_idx],
                self.coverage[topo_idx],
                self.freq_occurence[topo_idx],
                self.avg_lifespan[topo_idx],
            ] + self.transition_mat[topo_idx, :].tolist()
        if write_attrs:
            for key, val in self.attrs.items():
                df[key] = str(val)
        return df


def get_group_latent(subject_maps, decomposition_type, subject_channels):
    """
    Computes group-level latent representation from single subject ones and
    match them to order of microstate templates.

    :param subject_maps: list of individual subjects' latent maps
    type subject_maps: list[np.ndarray]
    :param decomposition_type: decomposition type
    :type decomposition_type: str
    :param subject_channels: list of all channel lists for all subjects
    :type subject_channels: list[list[str]]
    """
    assert decomposition_type in [
        "kmeans",
        "AAHC",
        "TAAHC",
        "PCA",
        "ICA",
        "hmm",
    ]
    assert len(subject_maps) == len(subject_channels)
    assert all(
        subject_maps[i].shape[1] == len(subject_channels[i])
        for i in range(len(subject_maps))
    )

    # find common channels in all subjects
    chans_all_subjects = reduce(np.intersect1d, subject_channels)
    subject_maps_same_chans = []
    logging.info(
        f"# of common channels for all subjects: {len(chans_all_subjects)}"
    )
    for subject in range(len(subject_channels)):
        # match channels
        _, idx_input, _ = np.intersect1d(
            subject_channels[subject],
            chans_all_subjects,
            return_indices=True,
        )
        subject_maps_same_chans.append(subject_maps[subject][:, idx_input])
    subject_maps = deepcopy(subject_maps_same_chans)

    shapes = set([subj_map.shape for subj_map in subject_maps])
    assert len(shapes) == 1, shapes
    no_states = subject_maps[0].shape[0]
    concat = np.concatenate(subject_maps, axis=0)
    if decomposition_type == "kmeans":
        group_mean, _, _, _ = segment_microstates(
            concat.T,
            method="mod_kmeans",
            n_states=no_states,
            use_gfp=False,
            return_polarity=False,
            n_inits=50,
        )
    elif decomposition_type == "AAHC":
        group_mean, _, _, _ = segment_microstates(
            concat.T,
            method="AAHC",
            n_states=no_states,
            use_gfp=False,
            return_polarity=False,
            n_inits=50,
        )
    elif decomposition_type == "TAAHC":
        group_mean, _, _, _ = segment_microstates(
            concat.T,
            method="TAAHC",
            n_states=no_states,
            use_gfp=False,
            return_polarity=False,
            n_inits=50,
        )
    elif decomposition_type == "PCA":
        pca = PCA(n_components=no_states)
        pca.fit(concat)
        group_mean = pca.components_.copy()
    elif decomposition_type == "ICA":
        ica = FastICA(n_components=no_states)
        ica.fit(concat)
        group_mean = ica.components_.copy()
    elif decomposition_type == "hmm":
        group_mean, _, _, _ = segment_hmm(
            concat.T,
            n_states=no_states,
            use_gfp=False,
            pca_preprocess=None,
            return_polarity=False,
        )
    ms_templates, channels_templates = load_Koenig_microstate_templates(
        n_states=no_states
    )
    # match channels
    _, idx_input, idx_sortby = np.intersect1d(
        chans_all_subjects,
        channels_templates,
        return_indices=True,
    )
    attribution, corrs_template = match_reorder_topomaps(
        group_mean[:, idx_input],
        ms_templates[:, idx_sortby],
        return_correlation=True,
        return_attribution_only=True,
    )
    return (
        group_mean[attribution, :],
        corrs_template,
        chans_all_subjects,
    )
