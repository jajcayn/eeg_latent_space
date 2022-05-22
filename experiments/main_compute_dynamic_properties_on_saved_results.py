"""
Compute dynamic properties and entropy-based measures on already computed latent
decompositions.
"""
import argparse
import logging
import os
import string
from glob import glob

import dynamic_properties as dynprop
import numpy as np
import pandas as pd
from utils import run_in_parallel, set_logger

K_MAX = 6
MAX_LAG = 100
BLOCK_SIZES = [500, 1000, 2500, 5000]
HURST_MIN_WINDOW = 0.256  # sec
HURST_MAX_WINDOW = 16.0  # sec


def _get_dynamic_props_per_subject(args):
    subject_id, folder, n_states, sampling_freq, log2 = args
    # load subject segmentation
    sequence = np.load(os.path.join(folder, f"{subject_id}.npz"))[
        "latent_segmentation"
    ]
    # basics
    empirical_dist = dynprop.empirical_distribution(sequence, n_states)
    trans_mat = dynprop.empirical_trans_mat(sequence, n_states)
    equilibrium_dist = dynprop.equilibrium_distribution(trans_mat)
    # entropy
    mixing_time = dynprop.mixing_time(trans_mat)
    entropy = dynprop.H_1(sequence, n_states, log2=log2)
    max_entropy = dynprop.max_entropy(n_states, log2=log2)
    ent_rate, excess_ent = dynprop.excess_entropy_rate(
        sequence, n_states, kmax=K_MAX, log2=log2
    )
    mc_ent_rate = dynprop.markov_chain_entropy_rate(
        empirical_dist, trans_mat, log2=log2
    )
    # AIF
    aif1 = dynprop.lagged_mutual_information(
        sequence,
        n_states,
        max_lag=MAX_LAG,
        log2=log2,
        pbar=False,
    )
    _, aif_1st_peak_time = dynprop.find_1st_aif_peak(aif1, sampling_freq)
    # Markovianity
    m0 = dynprop.test_markovianity_nth_order(
        sequence, n_states, order=0, log2=log2, detailed=True
    )
    m1 = dynprop.test_markovianity_nth_order(
        sequence, n_states, order=1, log2=log2, detailed=True
    )
    m2 = dynprop.test_markovianity_nth_order(
        sequence, n_states, order=2, log2=log2, detailed=True
    )
    # stationarity
    stationarity = {}
    for block_size in BLOCK_SIZES:
        stationarity[
            block_size
        ] = dynprop.test_stationarity_conditional_homogeneity(
            sequence, n_states, block_size, detailed=True
        )
    # Hurst
    H, c, _ = dynprop.estimate_hurst(
        sequence,
        HURST_MIN_WINDOW,
        HURST_MAX_WINDOW,
        sampling_freq,
        detailed=True,
    )

    # DataFrame
    topo_names = list(string.ascii_uppercase)[:n_states]
    df = pd.DataFrame(
        columns=[
            "subject_id",
            "latent map",
            "distribution",
            "eq. distribution",
            "mixing time",
            "entropy",
            "max entropy",
            "entropy rate",
            "excess entropy",
            "MC entropy rate",
            "AIF 1st peak [ms]",
            "Markovian 0th t",
            "Markovian 0th p-val",
            "Markovian 1th t",
            "Markovian 1th p-val",
            "Markovian 2th t",
            "Markovian 2th p-val",
        ]
        + [
            f"stationarity L={bs} {res}"
            for bs in BLOCK_SIZES
            for res in ["t", "p-val"]
        ]
        + ["Hurst exp.", "Hurst intercept"]
    )
    for topo_idx, topo_name in enumerate(topo_names):
        df.loc[topo_idx] = (
            [
                subject_id,
                topo_name,
                empirical_dist[topo_idx],
                equilibrium_dist[topo_idx],
                mixing_time,
                entropy,
                max_entropy,
                ent_rate,
                excess_ent,
                mc_ent_rate,
                aif_1st_peak_time,
                m0[2],
                m0[0],
                m1[2],
                m1[0],
                m2[2],
                m2[0],
            ]
            + [stationarity[bs][ndx] for bs in BLOCK_SIZES for ndx in [2, 0]]
            + [H, c]
        )

    return df


def main(input_glob, sampling_freq, log2):
    set_logger()
    for folder in sorted(glob(input_glob)):
        logging.info(f"Computing on {os.path.basename(folder)}")
        df_latent = pd.read_csv(os.path.join(folder, "latent_stats.csv"))
        subjects = list(df_latent["subject_id"].unique())
        n_states = df_latent["latent map"].nunique()

        all_dfs = run_in_parallel(
            _get_dynamic_props_per_subject,
            [
                (subject_id, folder, n_states, sampling_freq, log2)
                for subject_id in subjects
            ],
        )
        full_df = pd.concat(list(all_dfs), axis=0)
        full_df.to_csv(os.path.join(folder, "dynamic_stats.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute dynamic measures on computed latent decompositions"
    )
    parser.add_argument(
        "input_data_glob",
        type=str,
        help="Input data glob / folder paths, not files!",
    )
    parser.add_argument(
        "--data_sampling_freq",
        type=float,
        default=250.0,
        help="Sampling frequency of the data",
    )
    parser.add_argument(
        "--log2",
        action="store_true",
        default=False,
        help="Whether to compute all Shannon stuff with log2 or ln",
    )

    args = parser.parse_args()
    main(args.input_data_glob, args.data_sampling_freq, args.log2)
