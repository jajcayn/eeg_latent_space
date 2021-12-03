"""
Script for computing spectral differences in the EC/EO data.
"""
import argparse
import logging
import os
from copy import deepcopy
from glob import glob
from multiprocessing import cpu_count

import mne
import numpy as np
import xarray as xr
from eeg_recording import SingleSubjectRecording
from surrogates import SurrogateRecording
from utils import RESULTS_ROOT, make_dirs, run_in_parallel, set_logger, today

BANDS = [
    (0.0, 4.0, "Delta"),
    (4.0, 8.0, "Theta"),
    (8.0, 12.0, "Alpha"),
    (12.0, 30.0, "Beta"),
    (30.0, 45.0, "Gamma"),
]


def _compute_freq_stats_surrs(args):
    recording, bands, data_type, surr_type, surr_no = args
    recording.construct_surrogates(
        surrogate_type=surr_type, univariate=False, n_iterations=20
    )
    events, event_id = mne.events_from_annotations(recording._data)
    epochs = mne.Epochs(
        recording._data,
        events,
        event_id=event_id,
        event_repeated="merge",
        tmax=2.0,
        tmin=0.0,
        baseline=None,
        preload=True,
    )
    all_data = []
    for band in bands:
        filt_envelope = (
            deepcopy(epochs)
            .filter(band[0], band[1])
            .apply_hilbert(envelope=True)
        )
        # epochs x channels x time in epoch
        band_data = (
            xr.DataArray(
                filt_envelope.get_data() * 1e6,
                dims=["epochs", "channels", "time"],
                coords={
                    "epochs": np.arange(filt_envelope.get_data().shape[0]),
                    "channels": filt_envelope.info["ch_names"],
                    "time": np.arange(0, filt_envelope.get_data().shape[-1])
                    * (1.0 / filt_envelope.info["sfreq"]),
                },
            )
            .assign_coords(
                {
                    "band": band[2],
                    "surrogate": surr_no,
                    "data_type": data_type,
                    "subject_id": recording.subject_id.split("_")[0],
                }
            )
            .expand_dims(["band", "data_type", "subject_id", "surrogate"])
        )
        # mean over epochs
        band_data = band_data.mean(dim="epochs")
        all_data.append(band_data)

    return xr.concat(all_data, dim="band").stack(
        stacked=["data_type", "subject_id", "surrogate"]
    )


def _compute_freq_stats_data(args):
    recording, bands, data_type = args
    # create 2s Epochs
    events, event_id = mne.events_from_annotations(recording._data)
    epochs = mne.Epochs(
        recording._data,
        events,
        event_id=event_id,
        event_repeated="merge",
        tmax=2.0,
        tmin=0.0,
        baseline=None,
        preload=True,
    )
    all_data = []
    for band in bands:
        filt_envelope = (
            deepcopy(epochs)
            .filter(band[0], band[1])
            .apply_hilbert(envelope=True)
        )
        # epochs x channels x time in epoch
        band_data = (
            xr.DataArray(
                filt_envelope.get_data() * 1e6,
                dims=["epochs", "channels", "time"],
                coords={
                    "epochs": np.arange(filt_envelope.get_data().shape[0]),
                    "channels": filt_envelope.info["ch_names"],
                    "time": np.arange(0, filt_envelope.get_data().shape[-1])
                    * (1.0 / filt_envelope.info["sfreq"]),
                },
            )
            .assign_coords(
                {
                    "band": band[2],
                    "data_type": data_type,
                    "subject_id": recording.subject_id.split("_")[0],
                }
            )
            .expand_dims(["band", "data_type", "subject_id"])
        )
        # mean over epochs
        band_data = band_data.mean(dim="epochs")
        all_data.append(band_data)

    return xr.concat(all_data, dim="band").stack(
        stacked=["data_type", "subject_id"]
    )


def main(
    input_data, surr_type=None, num_surrs=0, workers=cpu_count(), time_avg=False
):
    surr_suffix = "" if surr_type is None else f"_{surr_type}"
    result_dir = os.path.join(
        RESULTS_ROOT, f"{today()}_EEG_ECvsEO_spectral_diffs{surr_suffix}"
    )
    make_dirs(result_dir)
    set_logger(log_filename=os.path.join(result_dir, "log"))
    logging.info("Loading subject data...")

    recordings_EC = []
    for data_file in sorted(glob(f"{input_data}/*_EC.set")):
        mne_data = mne.io.read_raw_eeglab(data_file, preload=True)
        subject_id = "-".join(os.path.basename(data_file).split(".")[:-1])
        recordings_EC.append(
            SingleSubjectRecording(subject_id=subject_id, data=mne_data)
        )
    logging.info(f"Loaded {len(recordings_EC)} EC data files")

    recordings_EO = []
    for data_file in sorted(glob(f"{input_data}/*_EO.set")):
        mne_data = mne.io.read_raw_eeglab(data_file, preload=True)
        subject_id = "-".join(os.path.basename(data_file).split(".")[:-1])
        recordings_EO.append(
            SingleSubjectRecording(subject_id=subject_id, data=mne_data)
        )
    logging.info(f"Loaded {len(recordings_EO)} EO data files")

    if surr_type is None:

        results_EC = run_in_parallel(
            _compute_freq_stats_data,
            [(deepcopy(recording), BANDS, "EC") for recording in recordings_EC],
            workers=workers,
        )

        results_EO = run_in_parallel(
            _compute_freq_stats_data,
            [(deepcopy(recording), BANDS, "EO") for recording in recordings_EO],
            workers=workers,
        )

    else:
        assert num_surrs > 0
        results_EC = run_in_parallel(
            _compute_freq_stats_surrs,
            [
                (
                    SurrogateRecording.from_data(recording),
                    BANDS,
                    "EC",
                    surr_type,
                    surr_no,
                )
                for recording in recordings_EC
                for surr_no in range(num_surrs)
            ],
            workers=workers,
        )

        results_EO = run_in_parallel(
            _compute_freq_stats_surrs,
            [
                (
                    SurrogateRecording.from_data(recording),
                    BANDS,
                    "EO",
                    surr_type,
                    surr_no,
                )
                for recording in recordings_EO
                for surr_no in range(num_surrs)
            ],
            workers=workers,
        )

    all_data = xr.concat(
        list(results_EC) + list(results_EO), dim="stacked"
    ).unstack("stacked")

    if time_avg:
        all_data = all_data.mean("time")
    all_data.to_netcdf(os.path.join(result_dir, "epochs_bands_envelopes.nc"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subjectwise latent EEG decomposition"
    )
    parser.add_argument(
        "input_data", type=str, help="Folder with data files [EEGLAB]"
    )
    parser.add_argument(
        "surrogate_type",
        type=str,
        help="Type of surrogate bootstrapping: `FT`, `AAFT`, `IAAFT`, "
        "`shuffle`",
        default=None,
    )
    parser.add_argument(
        "--num_surrogates", type=int, help="number of surrogates", default=0
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="number of processes to launch",
    )
    parser.add_argument(
        "--time_avg",
        help="whether to perform an average over time before saving",
        dest="time_avg",
        action="store_true",
    )
    args = parser.parse_args()
    main(
        args.input_data,
        args.surrogate_type,
        args.num_surrogates,
        args.workers,
        args.time_avg,
    )
