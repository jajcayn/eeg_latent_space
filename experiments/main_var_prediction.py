"""
Run VAR model on EEG data and try to see microstate predictions.
"""

import argparse
import logging
import os
import string
import warnings
from copy import deepcopy
from glob import glob
from multiprocessing import cpu_count

import dynamic_properties as dynprop
import mne
import numpy as np
import pandas as pd
import xarray as xr
from data_utils import load_Koenig_microstate_templates
from eeg_recording import SingleSubjectRecording
from plotting import plot_eeg_topomaps
from statsmodels.tsa.api import VAR
from tqdm import tqdm
from utils import RESULTS_ROOT, make_dirs, run_in_parallel, set_logger, today

MAX_VAR_ORDER = 10
SEGMENT_START = 1 * 60.0  # in seconds
NO_STATES = 4
DATA_FILTER = [2.0, 20.0]
USE_GFP = True
LOG2 = True
K_MAX = 6
MI_MAX_LAG = 100


def order_df(results):
    df = pd.DataFrame(results.summary().data)
    for col in df:
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace("*", "")
    df.columns = df.iloc[0]
    df = df.iloc[1:, 1:]
    df.index = np.arange(0, len(df))
    df.index.name = "order"
    df = df.astype(float)
    return df


def _estimate_order(args):
    file, segment_length = args
    subject_id = os.path.basename(file).split(".")[0]
    mne_data = mne.io.read_raw_eeglab(file, preload=True)
    mne_data.crop(tmin=SEGMENT_START, tmax=SEGMENT_START + segment_length)
    pd_data = pd.DataFrame(
        mne_data.get_data().T * 10e6,
        columns=mne_data.info["ch_names"],
        index=mne_data.times,
    )
    model = VAR(pd_data)
    order_results = model.select_order(MAX_VAR_ORDER)
    orders_df = order_df(order_results)
    return orders_df["AIC"].to_frame().rename(columns={"AIC": subject_id})


def _simulate_var(args):
    file, segment_length, final_order, var_length, results_folder = args
    subject_id = os.path.basename(file).split(".")[0]
    mne_data = mne.io.read_raw_eeglab(file, preload=True)
    mne_data.crop(tmin=SEGMENT_START, tmax=SEGMENT_START + segment_length)
    pd_data = pd.DataFrame(
        mne_data.get_data().T * 10e6,
        columns=mne_data.info["ch_names"],
        index=mne_data.times,
    )
    model = VAR(pd_data)
    fit_results = model.fit(final_order)
    simulated = fit_results.simulate_var(
        steps=int(var_length * mne_data.info["sfreq"])
    )
    simulated_mne = mne.io.RawArray(simulated.T / 10e6, mne_data.info)
    simulated_mne.save(
        os.path.join(results_folder, f"{subject_id}_var.fif"), overwrite=True
    )

    return (subject_id, simulated_mne)


def _compute_microstates(args):
    recording = args
    recording.preprocess(DATA_FILTER[0], DATA_FILTER[1])
    recording.run_latent_kmeans(n_states=NO_STATES, use_gfp=USE_GFP)
    ms_templates, channels_templates = load_Koenig_microstate_templates(
        n_states=NO_STATES
    )
    try:
        recording.match_reorder_segmentation(ms_templates, channels_templates)
        recording.compute_segmentation_stats()
        recording.attrs = {
            "no_states": NO_STATES,
            "filter": DATA_FILTER,
            "decomposition_type": "modified K-Means",
            "use_gfp": USE_GFP,
        }
        assert recording.computed_stats
        return recording

    except TypeError:
        return None


def _compute_microstates_var_segments(args):
    var_subject_id, var_segment, segment_length, tmax, results_folder = args
    # load var
    full_var = mne.io.read_raw_fif(
        os.path.join(results_folder, f"{var_subject_id}_var.fif"), preload=True
    )
    # crop
    cropped = full_var.crop(tmin=var_segment * segment_length, tmax=tmax)
    if (cropped.times[-1] - cropped.times[0]) < (segment_length - 0.1):
        return
    # compute microstates
    recording = SingleSubjectRecording(
        subject_id=var_subject_id + f"_VAR-{var_segment+1}-segment",
        data=cropped,
    )
    recording.preprocess(DATA_FILTER[0], DATA_FILTER[1])
    try:
        recording.run_latent_kmeans(n_states=NO_STATES, use_gfp=USE_GFP)
        ms_templates, channels_templates = load_Koenig_microstate_templates(
            n_states=NO_STATES
        )
        recording.match_reorder_segmentation(ms_templates, channels_templates)
        recording.compute_segmentation_stats()
        recording.attrs = {
            "no_states": NO_STATES,
            "filter": DATA_FILTER,
            "decomposition_type": "modified K-Means",
            "use_gfp": USE_GFP,
        }
        assert recording.computed_stats
        return recording
    except (TypeError, ValueError):
        return None


def _save_recordings(args):
    recording, result_dir = args
    recording.save_latent(path=result_dir)
    plot_eeg_topomaps(
        recording.latent_maps,
        recording.info,
        xlabels=[
            f"r={np.abs(corr):.3f} vs. template"
            for corr in recording.corrs_template
        ],
        tit=recording.subject_id,
        fname=os.path.join(result_dir, f"{recording.subject_id}_topo.png"),
        transparent=True,
    )


def _compute_dynamical_stats(args):
    sequence, sampling_freq, subject_id = args
    empirical_dist = dynprop.empirical_distribution(sequence, NO_STATES)
    trans_mat = dynprop.empirical_trans_mat(sequence, NO_STATES)
    mixing_time = dynprop.mixing_time(trans_mat)
    entropy = dynprop.H_1(sequence, NO_STATES, log2=LOG2)
    max_entropy = dynprop.max_entropy(NO_STATES, log2=LOG2)
    ent_rate, excess_ent = dynprop.excess_entropy_rate(
        sequence, NO_STATES, kmax=K_MAX, log2=LOG2
    )
    mc_ent_rate = dynprop.markov_chain_entropy_rate(
        empirical_dist, trans_mat, log2=LOG2
    )
    aif1 = dynprop.lagged_mutual_information(
        sequence,
        NO_STATES,
        max_lag=MI_MAX_LAG,
        log2=LOG2,
        pbar=False,
    )
    _, aif_1st_peak_time = dynprop.find_1st_aif_peak(aif1, sampling_freq)

    return pd.DataFrame(
        {
            "subject_id": subject_id,
            "mixing time": mixing_time,
            "entropy": entropy,
            "max entropy": max_entropy,
            "entropy_rate": ent_rate,
            "MC entropy rate": mc_ent_rate,
            "AIF 1st peak": aif_1st_peak_time,
        },
        index=[0],
    )


def main(
    input_data,
    no_random_subjects,
    segment_length,
    var_total_length=30 * 60.0,
    n_samples_var_segments=np.inf,
    data_type="EC",
    workers=cpu_count(),
    save_all=False,
):
    mne.set_log_level("error")
    warnings.filterwarnings("ignore")

    result_dir = os.path.join(
        RESULTS_ROOT,
        f"{today()}_VARprocess_{data_type}_{no_random_subjects}subjects_"
        f"{segment_length}s_segment_{var_total_length}s_VARlength",
    )
    make_dirs(result_dir)
    set_logger(log_filename=os.path.join(result_dir, "log"))

    logging.info(f"Selecting {no_random_subjects} subjects at random...")
    all_files = glob(f"{input_data}/*_{data_type}.set")
    if no_random_subjects == "all":
        chosen_files = all_files
        no_random_subjects = len(all_files)
    else:
        no_random_subjects = int(no_random_subjects)

        chosen_files = sorted(
            np.random.choice(
                all_files, size=no_random_subjects, replace=False
            ).tolist()
        )
        assert len(chosen_files) == no_random_subjects

    logging.info("Estimating VAR order...")
    all_orders = run_in_parallel(
        _estimate_order,
        [(file, segment_length) for file in chosen_files],
        workers=workers,
    )
    all_orders = pd.concat(all_orders)
    all_orders.to_csv(os.path.join(result_dir, "VAR_orders_aic.csv"))

    min_order = all_orders.idxmin()
    final_order = int(min_order.median())
    logging.info(f"Ideal VAR order seems to be {final_order}")

    logging.info(f"Simulating VAR of order {final_order} per subject...")
    simulated_results = run_in_parallel(
        _simulate_var,
        [
            (file, segment_length, final_order, var_total_length, result_dir)
            for file in chosen_files
        ],
        workers=workers,
    )
    logging.info("All done and saved.")

    logging.info("Computing microstates on everything...")
    full_data_recordings = []
    logging.info("Adding real data 1st segment...")
    # real data first segment
    for file in tqdm(chosen_files):
        subject_id = os.path.basename(file).split(".")[0]
        subject_id += "_1st-segment"
        mne_data = mne.io.read_raw_eeglab(file, preload=True)
        mne_data.crop(tmin=SEGMENT_START, tmax=SEGMENT_START + segment_length)
        full_data_recordings.append(
            SingleSubjectRecording(subject_id=subject_id, data=mne_data)
        )
    # real data second segment
    logging.info("Adding real data 2nd segment...")
    for file in tqdm(chosen_files):
        subject_id = os.path.basename(file).split(".")[0]
        subject_id += "_2nd-segment"
        mne_data = mne.io.read_raw_eeglab(file, preload=True)
        mne_data.crop(
            tmin=SEGMENT_START + segment_length,
            tmax=SEGMENT_START + 2 * segment_length,
        )
        full_data_recordings.append(
            SingleSubjectRecording(subject_id=subject_id, data=mne_data)
        )
    # VAR data total
    logging.info("Adding VAR data in full...")
    for (var_subject_id, var_data) in simulated_results:
        # full VAR simulation
        full_data_recordings.append(
            SingleSubjectRecording(
                subject_id=var_subject_id + "_VAR-full", data=deepcopy(var_data)
            )
        )

    assert len(full_data_recordings) == (no_random_subjects * 3), (
        len(full_data_recordings),
        no_random_subjects,
    )
    logging.info("Computing microstates on full data...")
    full_microstate_results = run_in_parallel(
        _compute_microstates,
        [recording for recording in full_data_recordings],
        workers=workers,
    )
    del full_data_recordings

    # per segment VAR simulation
    logging.info("Adding VAR data by segments...")
    n_var_segments = int(np.ceil(var_total_length / segment_length))
    sampling_size = (
        n_var_segments
        if np.isinf(n_samples_var_segments)
        else n_samples_var_segments
    )
    assert sampling_size <= n_var_segments
    sampled_segments = sorted(
        np.random.choice(n_var_segments, size=sampling_size, replace=False)
    )
    segment_data_for_ms = []
    for (var_subject_id, _) in simulated_results:
        for var_segment in sampled_segments:
            if var_segment + 1 == n_var_segments:
                tmax = None
            else:
                tmax = (var_segment + 1) * segment_length
            segment_data_for_ms.append(
                (
                    var_subject_id,
                    var_segment,
                    segment_length,
                    tmax,
                    result_dir,
                )
            )

    del simulated_results
    logging.info("Computing microstates on VAR segments...")
    segment_microstates_results = run_in_parallel(
        _compute_microstates_var_segments,
        segment_data_for_ms,
        workers=workers,
    )
    segment_microstates_results = [
        result for result in segment_microstates_results if result
    ]
    del segment_data_for_ms

    microstate_results = full_microstate_results + segment_microstates_results
    microstate_results = [res for res in microstate_results if res is not None]
    logging.info("Microstates done.")
    if save_all:
        logging.info("Saving stuff...")
        run_in_parallel(
            _save_recordings,
            [(recording, result_dir) for recording in microstate_results],
        )

    logging.info("Saving maps into netcdf...")
    all_topomaps = []
    subject_ids = []
    types = []
    for recording in microstate_results:
        subject_ids.append(recording.subject_id.split("_")[0])
        types.append(recording.subject_id.split("_")[2])
        all_topomaps.append(recording.latent_maps)
    array_topomaps = np.empty(
        (
            len(subject_ids),
            NO_STATES,
            max([arr.shape[1] for arr in all_topomaps]),
        )
    )
    array_topomaps[:] = np.nan
    for i, arr in enumerate(all_topomaps):
        array_topomaps[i, :, : arr.shape[1]] = arr
    topo_maps = xr.DataArray(
        array_topomaps,
        dims=["stack", "latent map", "channels"],
        coords={
            "subject_id": ("stack", subject_ids),
            "type": ("stack", types),
            "latent map": list(string.ascii_uppercase)[:NO_STATES],
            "channels": np.arange(array_topomaps.shape[2]),
        },
    )
    # do this later
    # topo_maps = topo_maps.set_index({"stack": ["subject_id", "type"]})
    topo_maps.to_netcdf(os.path.join(result_dir, "topomaps.nc"))

    logging.info("Saving microstate statistics to csv...")
    full_df = pd.concat(
        [
            recording.get_stats_pandas(write_attrs=True)
            for recording in microstate_results
        ],
        axis=0,
    )
    full_df.to_csv(os.path.join(result_dir, "static_stats.csv"))

    logging.info("Computing dynamical stats...")
    dyn_stats = run_in_parallel(
        _compute_dynamical_stats,
        [
            (recording.latent_segmentation, 250.0, recording.subject_id)
            for recording in microstate_results
        ],
        workers=workers,
    )
    dyn_stats = pd.concat(dyn_stats)
    dyn_stats.to_csv(os.path.join(result_dir, "dynamic_stats.csv"))

    if not save_all:
        logging.info("Removing fif files...")
        for file in glob(f"{result_dir}/*.fif"):
            os.remove(file)

    logging.info("All done, bye.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VAR modelling of latent EEG decomposition"
    )
    parser.add_argument(
        "input_data", type=str, help="Folder with data files [EEGLAB]"
    )
    parser.add_argument(
        "no_random_subjects",
        type=str,
        help="Number of random subjects to select.",
    )
    parser.add_argument(
        "segment_length",
        type=float,
        help="Segment lengths in sec for data and VAR process.",
    )
    parser.add_argument(
        "--var_total_length",
        type=float,
        help="Total length of simulated VAR.",
        default=30 * 60.0,
    )
    parser.add_argument(
        "--n_samples_var_segments",
        type=int,
        help="Whether to sample VAR segments to save memory",
        default=np.inf,
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="EC",
        choices=["EC", "EO"],
        help="data type: EC vs. EO",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="number of processes to launch",
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        default=False,
        help="Whether to store all the stuff",
    )
    args = parser.parse_args()
    main(
        args.input_data,
        args.no_random_subjects,
        args.segment_length,
        args.var_total_length,
        args.n_samples_var_segments,
        args.data_type,
        args.workers,
        args.save_all,
    )
