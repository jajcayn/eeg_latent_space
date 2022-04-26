"""
Script for running HMM model on concatenated data, instead of subjectwise.
"""
import argparse
import logging
import os
from copy import deepcopy
from glob import glob
from multiprocessing import cpu_count

import mne
import numpy as np
import pandas as pd
from data_utils import (
    corr_vectors,
    get_gfp_peaks,
    load_Koenig_microstate_templates,
)
from eeg_recording import SingleSubjectRecording
from hmmlearn.hmm import GaussianHMM
from plotting import plot_eeg_topomaps
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from surrogates import SurrogateRecording
from utils import RESULTS_ROOT, make_dirs, run_in_parallel, set_logger, today


def _preprocess(args):
    recording, surr_type, data_filter, n_pca_comps, use_gfp, seed = args
    if surr_type is not None:
        # first, construct surrogates
        recording.construct_surrogates(
            surrogate_type=surr_type,
            univariate=False,
            n_iterations=20,
            seed=seed,
        )
    recording.preprocess(data_filter[0], data_filter[1])
    recording._data.apply_hilbert(envelope=True)
    data_numpy = recording.data.copy()

    pca = PCA(n_components=n_pca_comps)
    train_data = pca.fit_transform(data_numpy.T).T
    if use_gfp:
        (peaks, gfp_curve) = get_gfp_peaks(train_data)
    else:
        peaks = np.arange(train_data.shape[1])
        gfp_curve = np.std(train_data, axis=0)
    return train_data[:, peaks], train_data, peaks, gfp_curve, recording


def _postprocess(args):
    (
        subject_decoding,
        recording,
        peaks,
        gfp_curve,
        no_states,
        use_gfp,
        data_filter,
    ) = args
    oh_enc = OneHotEncoder(sparse=False, categories=[np.arange(no_states)])
    one_hot = oh_enc.fit_transform(subject_decoding[:, np.newaxis])
    # get HMM maps via linear regression
    lin_reg = LinearRegression(fit_intercept=False)
    scaler = StandardScaler()
    orig_data_scaled = scaler.fit_transform(recording.data.T)
    lin_reg.fit(one_hot, orig_data_scaled)

    maps = deepcopy(lin_reg.coef_.T)
    segmentation = deepcopy(subject_decoding)
    activation = maps.dot(recording.data)
    polarity = np.sign(np.choose(segmentation, activation))

    gfp_sum_sq = np.sum(gfp_curve**2)
    peaks_sum_sq = np.sum(recording.data[:, peaks].std(axis=0) ** 2)

    map_corr = corr_vectors(recording.data, maps[segmentation].T)
    gfp_corr = corr_vectors(
        recording.data[:, peaks], maps[segmentation[peaks]].T
    )
    gev = sum((gfp_curve * map_corr) ** 2) / gfp_sum_sq
    gev_gfp = (
        sum((recording.data[:, peaks].std(axis=0) * gfp_corr) ** 2)
        / peaks_sum_sq
    )
    recording.latent_maps = maps
    recording.latent_segmentation = segmentation
    recording.polarity = polarity
    recording.gev_tot = gev
    recording.gev_gfp = gev_gfp
    recording.n_states = no_states

    ms_templates, channels_templates = load_Koenig_microstate_templates(
        n_states=no_states
    )
    recording.match_reorder_segmentation(ms_templates, channels_templates)
    recording.compute_segmentation_stats()
    recording.attrs = {
        "no_states": no_states,
        "filter": data_filter,
        "decomposition_type": "HMMconcat",
        "use_gfp": use_gfp,
    }
    assert recording.computed_stats
    return recording


def main(
    input_data,
    surr_type,
    no_states=4,
    data_filter=(2.0, 20.0),
    n_pca_comps=20,
    data_type="EC",
    use_gfp=True,
    crop=None,
    workers=cpu_count(),
    seed=42,
):
    crop_str = f"crop{crop}s_" if crop is not None else ""
    if surr_type is None:
        result_dir = os.path.join(
            RESULTS_ROOT,
            f"{today()}_{no_states}HMMconcat_{data_filter[0]}-{data_filter[1]}"
            f"Hz_{data_type}_{crop_str}subjectwise",
        )
    else:
        result_dir = os.path.join(
            RESULTS_ROOT,
            f"{today()}_{surr_type.upper()}surrs_{no_states}HMMconcat_"
            f"{data_filter[0]}-{data_filter[1]}Hz_{data_type}_seeded_{crop_str}"
            "subjectwise",
        )
    make_dirs(result_dir)
    set_logger(log_filename=os.path.join(result_dir, "log"))
    logging.info("Loading subject data...")
    recordings = []
    for data_file in sorted(glob(f"{input_data}/*_{data_type}.set")):
        mne_data = mne.io.read_raw_eeglab(data_file, preload=True)
        if crop is not None:
            mne_data.crop(tmax=crop)
        subject_id = "-".join(os.path.basename(data_file).split(".")[:-1])
        eeg = SingleSubjectRecording(subject_id=subject_id, data=mne_data)
        if surr_type is not None:
            eeg = SurrogateRecording.from_data(eeg)
        recordings.append(eeg)
    logging.info(f"Loaded {len(recordings)} data files")
    logging.info("Computing HMMconcat decomposition per subject...")

    preprocessed_recordings = run_in_parallel(
        _preprocess,
        [
            (
                deepcopy(recording),
                surr_type,
                data_filter,
                n_pca_comps,
                use_gfp,
                seed,
            )
            for recording in recordings
        ],
        workers=workers,
    )
    concat_timeseries_train = np.concatenate(
        [res[0] for res in preprocessed_recordings], axis=1
    )
    concat_timeseries_full = np.concatenate(
        [res[1] for res in preprocessed_recordings], axis=1
    )
    # fit HMM
    model = GaussianHMM(
        n_components=no_states,
        covariance_type="full",
        n_iter=200,
        tol=1e-6,
        verbose=False,
    )
    model.fit(concat_timeseries_train.T)
    # decode using Viterbi
    _, decoding = model.decode(concat_timeseries_full.T, algorithm="viterbi")

    postprocess_args = []
    idx_start = 0
    for result in preprocessed_recordings:
        _, _, peaks, gfp_curve, recording = result
        # new index for end = start + length
        idx_end = idx_start + recording.data.shape[1]
        postprocess_args.append(
            (
                deepcopy(decoding)[idx_start:idx_end],
                recording,
                peaks,
                gfp_curve,
                no_states,
                use_gfp,
                data_filter,
            )
        )
        # new index for start = end now
        idx_start = idx_end

    results = run_in_parallel(_postprocess, postprocess_args, workers=workers)

    results = [res for res in results if res is not None]
    logging.info("Latent decomposition done.")
    for recording in results:
        recording.save_latent(path=result_dir)
        title = f"{recording.subject_id} ~ HMMconcat: {data_filter[0]}-"
        f"{data_filter[1]} Hz"
        plot_eeg_topomaps(
            recording.latent_maps,
            recording.info,
            xlabels=[
                f"r={np.abs(corr):.3f} vs. template"
                for corr in recording.corrs_template
            ],
            tit=title,
            fname=os.path.join(
                result_dir, f"{recording.subject_id}_ind_topo.png"
            ),
            transparent=True,
        )

    logging.info("Saving latent decomposition statistics to csv...")
    full_df = pd.concat(
        [recording.get_stats_pandas(write_attrs=True) for recording in results],
        axis=0,
    )
    full_df.to_csv(os.path.join(result_dir, "latent_stats.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HMM-latent EEG decomposition with concatenated subjectws"
    )
    parser.add_argument(
        "input_data", type=str, help="Folder with data files [EEGLAB]"
    )
    parser.add_argument(
        "--surrogate_type",
        type=str,
        default=None,
        help="Type of surrogate bootstrapping: `FT`, `AAFT`, `IAAFT`, "
        "`shuffle`",
    )
    parser.add_argument(
        "--no_states", type=int, default=4, help="number of latent states"
    )
    parser.add_argument(
        "--filter",
        type=float,
        nargs="+",
        default=[2.0, 20.0],
        help="bandpass filter to use",
    )
    parser.add_argument(
        "--n_pca_comps",
        type=int,
        default=20,
        help="number of PCA components to use for HMM fitting",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="EC",
        choices=["EC", "EO"],
        help="data type: EC vs. EO",
    )
    parser.add_argument("--use_gfp", action="store_true", default=True)
    parser.add_argument(
        "--crop",
        type=float,
        default=None,
        help="whether to crop data before computation, None for no cropping "
        "float in seconds for cropping",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="number of processes to launch",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(
        args.input_data,
        args.surrogate_type,
        args.no_states,
        args.filter,
        args.n_pca_comps,
        args.data_type,
        args.use_gfp,
        args.crop,
        args.workers,
        args.seed,
    )
