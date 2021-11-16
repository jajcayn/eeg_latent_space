"""
Script for running latent decomposition for surrogate data.
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
from data_utils import load_Koenig_microstate_templates
from eeg_recording import SingleSubjectRecording, get_group_latent
from plotting import plot_eeg_topomaps
from surrogates import SurrogateRecording
from utils import RESULTS_ROOT, make_dirs, run_in_parallel, set_logger, today


def _compute_latent(args):
    recording, d_type, surr_type, no_states, data_filter, use_gfp = args
    # first, construct surrogates
    recording.construct_surrogates(
        surrogate_type=surr_type, univariate=False, n_iterations=20
    )
    # now preprocess and run latent
    recording.preprocess(data_filter[0], data_filter[1])
    if d_type == "microstates":
        recording.run_latent_microstates(n_states=no_states, use_gfp=use_gfp)
    elif d_type == "PCA":
        recording.run_latent_pca(n_states=no_states, use_gfp=use_gfp)
    elif d_type == "ICA":
        recording.run_latent_ica(n_states=no_states, use_gfp=use_gfp)
    elif d_type == "hmm":
        recording.run_latent_hmm(
            n_states=no_states,
            use_gfp=use_gfp,
            pca_preprocess=0.9,
            envelope=True,
        )
    else:
        raise NotImplementedError("Unknown latent method")
    ms_templates, channels_templates = load_Koenig_microstate_templates(
        n_states=no_states
    )
    recording.match_reorder_segmentation(ms_templates, channels_templates)
    recording.reassign_segmentation_by_midpoints()
    recording.compute_segmentation_stats()
    recording.attrs = {
        "no_states": no_states,
        "filter": data_filter,
        "surrogate_type": surr_type,
        "decomposition_type": d_type,
        "use_gfp": use_gfp,
    }
    assert recording.computed_stats
    return recording


def main(
    input_data,
    decomp_type,
    surr_type,
    no_states=4,
    data_filter=(2.0, 20.0),
    data_type="EC",
    use_gfp=True,
    workers=cpu_count(),
):

    result_dir = os.path.join(
        RESULTS_ROOT,
        f"{today()}_{surr_type.upper()}surrs_{no_states}{decomp_type}_"
        f"{data_filter[0]}-{data_filter[1]}Hz_{data_type}_subjectwise",
    )
    make_dirs(result_dir)
    set_logger(log_filename=os.path.join(result_dir, "log"))
    logging.info("Loading subject data...")
    recordings = []
    for data_file in sorted(glob(f"{input_data}/*_{data_type}.set")):
        mne_data = mne.io.read_raw_eeglab(data_file, preload=True)
        subject_id = "-".join(os.path.basename(data_file).split(".")[:-1])
        orig_eeg = SingleSubjectRecording(subject_id=subject_id, data=mne_data)
        recordings.append(SurrogateRecording.from_data(orig_eeg))
    logging.info(f"Loaded {len(recordings)} data files")
    logging.info(f"Computing {decomp_type} decomposition per subject...")
    results = run_in_parallel(
        _compute_latent,
        [
            (
                deepcopy(recording),
                decomp_type,
                surr_type,
                no_states,
                data_filter,
                use_gfp,
            )
            for recording in recordings
        ],
        workers=workers,
    )
    logging.info("Latent decomposition done.")
    for recording in results:
        recording.save_latent(path=result_dir)
        title = f"{recording.subject_id} ~ {surr_type} x {decomp_type}: "
        f"{data_filter[0]}-{data_filter[1]} Hz"
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

    logging.info("Computing group mean topomaps...")
    group_mean, corrs_w_template, group_channels = get_group_latent(
        [recording.latent_maps for recording in results],
        decomposition_type=decomp_type,
        subject_channels=[recording.info["ch_names"] for recording in results],
    )
    np.savez(
        os.path.join(result_dir, "group_mean.npz"),
        latent_maps=group_mean,
        corrs_w_template=corrs_w_template,
        group_channels=group_channels,
    )
    max_chan_idx = np.array(
        [len(recording.info["ch_names"]) for recording in results]
    ).argmax()
    plot_eeg_topomaps(
        group_mean,
        results[max_chan_idx].info.pick_channels(group_channels, ordered=True),
        xlabels=[
            f"r={np.abs(corr):.3f} vs. template" for corr in corrs_w_template
        ],
        tit=title,
        fname=os.path.join(result_dir, "group_mean_topo.png"),
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
        description="Subjectwise latent EEG decomposition"
    )
    parser.add_argument(
        "input_data", type=str, help="Folder with data files [EEGLAB]"
    )
    parser.add_argument(
        "decomposition_type",
        type=str,
        help="Type of decomposition: `microstates`, `PCA`, `ICA` or `hmm`",
    )
    parser.add_argument(
        "surrogate_type",
        type=str,
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
        "--data_type",
        type=str,
        default="EC",
        choices=["EC", "EO"],
        help="data type: EC vs. EO",
    )
    parser.add_argument("--use_gfp", action="store_true", default=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="number of processes to launch",
    )

    args = parser.parse_args()
    main(
        args.input_data,
        args.decomposition_type,
        args.surrogate_type,
        args.no_states,
        args.filter,
        args.data_type,
        args.use_gfp,
        args.workers,
    )
