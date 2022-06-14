"""
Generate random EEG data as multivariate Gaussian and run the stuff.
"""
import argparse
import os
import string
from copy import deepcopy
from multiprocessing import cpu_count

import dynamic_properties as dynprop
import mne
import numpy as np
import pandas as pd
import xarray as xr
from eeg_recording import SingleSubjectRecording
from utils import RESULTS_ROOT, make_dirs, run_in_parallel, today

SAMPLING_RATE = 250.0  # Hz

DECOMPOSITION_FUNCTIONS = {
    "PCA": lambda x, ns: x.run_latent_pca(ns, use_gfp=True),
    "ICA": lambda x, ns: x.run_latent_ica(ns, use_gfp=True),
    "kmeans": lambda x, ns: x.run_latent_kmeans(ns, use_gfp=True),
    "AAHC": lambda x, ns: x.run_latent_aahc(ns, use_gfp=True),
    "TAAHC": lambda x, ns: x.run_latent_taahc(ns, use_gfp=True),
    "HMM": lambda x, ns: x.run_latent_hmm(ns, use_gfp=True, envelope=False),
}


def get_random_gaussian_data(
    n_channels, length, sampling_rate, avg_reference=True, seed=None
):
    """
    Get Gaussian data of length with n_channels. Data are sampled from
    multivariate normal distribution with random mean in N(0, 1) and random
    covariance computed as cov = A dot A' with A random matrix with elements
    from N(0, 1).
    """
    np.random.seed(seed)
    random_cov = np.random.normal(0.0, 1.0, size=(n_channels, n_channels))
    data = pd.DataFrame(
        np.random.multivariate_normal(
            # mean=np.random.normal(0.0, 1.0, size=n_channels),
            mean=[0.0] * n_channels,
            cov=np.dot(random_cov, random_cov.transpose()),
            size=(int(length * sampling_rate)),
        ),
        columns=[f"chan {i}" for i in range(n_channels)],
        index=np.arange(0, length, 1 / sampling_rate),
    )
    if avg_reference:
        data = data.subtract(data.mean(axis=1), axis=0)

    return data


def _process_recording(args):
    recording, n_states, n_channels, length = args
    all_stats = pd.DataFrame()
    dyn_stats = pd.DataFrame()
    maps = []
    orig_data = pd.DataFrame(
        recording.data.T,
        columns=[f"chan_{i}" for i in range(n_channels)],
        index=np.arange(0, length, 1 / SAMPLING_RATE),
    )
    orig_data["subject"] = recording.subject_id
    for method, func in DECOMPOSITION_FUNCTIONS.items():
        func(recording, n_states)
        orig_data[method] = recording.latent_segmentation
        maps.append(
            xr.DataArray(
                recording.latent_maps,
                dims=["latent map", "channels"],
                coords={
                    "latent map": list(string.ascii_uppercase)[:n_states],
                    "channels": [f"chan_{i}" for i in range(n_channels)],
                },
            )
            .assign_coords(
                {"algorithm": method, "subject": recording.subject_id}
            )
            .expand_dims(["algorithm", "subject"])
        )
        recording.compute_segmentation_stats()
        df = recording.get_stats_pandas().copy()
        df["algorithm"] = method
        all_stats = pd.concat([all_stats, df], axis=0)

        dyn_stats.loc[method, "subject"] = recording.subject_id
        dyn_stats.loc[method, "mixing time"] = dynprop.mixing_time(
            dynprop.empirical_trans_mat(recording.latent_segmentation, n_states)
        )
        dyn_stats.loc[method, "entropy"] = dynprop.H_1(
            recording.latent_segmentation, n_states, log2=True
        )
        dyn_stats.loc[method, "max entropy"] = dynprop.max_entropy(
            n_states, log2=True
        )
        dyn_stats.loc[method, "entropy rate"] = dynprop.excess_entropy_rate(
            recording.latent_segmentation, n_states, kmax=6, log2=True
        )[0]
        dyn_stats.loc[
            method, "MC entropy rate"
        ] = dynprop.markov_chain_entropy_rate(
            dynprop.empirical_distribution(
                recording.latent_segmentation, n_states
            ),
            dynprop.empirical_trans_mat(
                recording.latent_segmentation, n_states
            ),
            log2=True,
        )
        aif1 = dynprop.lagged_mutual_information(
            recording.latent_segmentation,
            n_states,
            max_lag=100,
            log2=True,
            pbar=False,
        )
        dyn_stats.loc[method, "AIF 1st peak"] = dynprop.find_1st_aif_peak(
            aif1, SAMPLING_RATE
        )[1]

    return (
        orig_data.reset_index()
        .rename(columns={"index": "time"})
        .set_index(["subject", "time"]),
        all_stats,
        dyn_stats.reset_index().rename(columns={"index": "algorithm"}),
        maps,
    )


def main(n_subjects, ts_length, n_channels, n_states, workers):
    result_dir = os.path.join(
        RESULTS_ROOT,
        f"{today()}_random_data_stuff_{n_subjects}subjects_{n_channels}"
        f"channels_{n_states}states",
    )
    make_dirs(result_dir)
    recordings = []
    for i in range(n_subjects):
        data = get_random_gaussian_data(
            n_channels, ts_length, SAMPLING_RATE, avg_reference=True
        )
        info = mne.create_info(
            ch_names=list(data.columns),
            ch_types=["eeg"] * n_channels,
            sfreq=SAMPLING_RATE,
        )
        mne_data = mne.io.RawArray(data.values.T, info)

        recordings.append(
            SingleSubjectRecording(subject_id=f"rnd_subject_{i}", data=mne_data)
        )

    results = run_in_parallel(
        _process_recording,
        [
            (deepcopy(recording), n_states, n_channels, ts_length)
            for recording in recordings
        ],
        workers=workers,
    )
    all_data = pd.concat([res[0] for res in results], axis=0)
    all_data.to_csv(os.path.join(result_dir, "data_and_segmentation.csv"))

    stats = pd.concat([res[1] for res in results], axis=0)
    stats.to_csv(os.path.join(result_dir, "basic_stats.csv"))

    dyn_stats = pd.concat([res[2] for res in results], axis=0)
    dyn_stats.to_csv(os.path.join(result_dir, "dyn_stats.csv"))

    maps = xr.combine_by_coords(sum([res[3] for res in results], []))
    maps.to_netcdf(os.path.join(result_dir, "topomaps.nc"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random multivariate pipeline")
    parser.add_argument("--n_subjects", type=int, default=50)
    parser.add_argument("--ts_length", type=float, default=10.0)
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--n_states", type=int, default=2)
    parser.add_argument("--workers", type=int, default=cpu_count())
    args = parser.parse_args()
    main(
        args.n_subjects,
        args.ts_length,
        args.n_channels,
        args.n_states,
        args.workers,
    )
